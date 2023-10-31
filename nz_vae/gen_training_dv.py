#!/usr/bin/env python
# coding: utf-8
import cosmosis
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import sys
import os


def setup_pipeline(sacc_file):
    ini = cosmosis.Inifile("data/params.ini")
    values = cosmosis.Inifile(ini["pipeline", "values"])

    ini["sacc_nz", "nz_file"] = sacc_file
    ini["lae_sacc_like", "data_file"] = sacc_file
    ini["camb", "feedback"] = 0
    values.remove_section("wl_photoz_errors")

    cosmosis.logs.set_verbosity("quiet")

    pipe = cosmosis.LikelihoodPipeline(ini, values=values)
    return pipe


def override_nz(pipe, z, nz):
    names = [p.name for p in pipe.modules]
    index = names.index("sacc_nz")
    module = pipe.modules[index]
    z = z.copy()
    nz = nz.copy()
    new_data = (z, {i: nz_i for i, nz_i in enumerate(nz)})
    module.data['nz_source'] = new_data



def override_likelihood(pipe):
    names = [p.name for p in pipe.modules]
    index = names.index("lae_sacc_like")
    module = pipe.modules[index]

    mean_nz_results = pipe.run_results(pipe.start_vector())

    y0 = mean_nz_results.block["data_vector", "2pt_theory"]

    module.data.data_y[:] = y0
    # module.data.cov[:, :] = orig_covmat
    # module.data.inv_cov[:, :] = orig_precmat
    sigma = module.data.cov[:, :].diagonal() ** 0.5

    return y0, sigma, module.data.inv_cov


def get_theory_vector(pipe, z, nz):
    override_nz(pipe, z, nz)
    r = pipe.run_results(pipe.start_vector())
    y = r.block["data_vector", "2pt_theory"]
    chi2 = r.block["data_vector", "2pt_chi2"]

    mu = r.block["data_vector", "2pt_data"]
    sigma = r.block["data_vector", "2pt_covariance"].diagonal() ** 0.5
    x = np.arange(sigma.size)
    return y, r.like, chi2


def load_nz_data(nz_realization_z_file, nz_realization_file):
    nz_data = np.load(nz_realization_file)['arr_0']
    z = np.load(nz_realization_z_file)['arr_0']
    nsample_total = nz_data.shape[0]
    ntomo = nz_data.shape[1]
    for i in range(nsample_total):
        for j in range(ntomo):
            norm = scipy.integrate.trapz(nz_data[i, j], z)
            nz_data[i, j] /= norm
    return z, nz_data


def main(nz_realization_z_file, nz_realization_file, data_vector_file, sacc_file, nsample, comm):
    import mpi4py.MPI

    rank = comm.rank
    size = comm.size

    pipe = setup_pipeline(sacc_file)

    z, nz_data = load_nz_data(nz_realization_z_file, nz_realization_file)

    override_nz(pipe, z, nz_data.mean(0))
    y0, sigma, precision_matrix = override_likelihood(pipe)
    ndata = len(y0)

    theory_vectors = np.zeros((nsample, ndata))
    likes = np.zeros(nsample)
    chi2 = np.zeros(nsample)

    for i in range(nsample):
        if i % size == rank:
            theory_vectors[i], likes[i], chi2[i] = get_theory_vector(
                pipe, z, nz_data[i]
            )
            print(rank, i, chi2[i])
            sys.stdout.flush()

    if rank == 0:
        comm.Reduce(mpi4py.MPI.IN_PLACE, theory_vectors)
        comm.Reduce(mpi4py.MPI.IN_PLACE, likes)
        comm.Reduce(mpi4py.MPI.IN_PLACE, chi2)
    else:
        comm.Reduce(theory_vectors, None)
        comm.Reduce(likes, None)
        comm.Reduce(chi2, None)

    if rank == 0:
        data = {
            "theory_vectors": theory_vectors,
            "likes": likes,
            "chi2": chi2,
            "y0": y0,
            "sigma": sigma,
            "precision_matrix": precision_matrix,
            "z": z,
        }
        np.savez(data_vector_file, data)


if __name__ == "__main__":
    import mpi4py.MPI

    main()
