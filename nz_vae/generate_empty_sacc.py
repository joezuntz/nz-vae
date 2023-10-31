# script to create sacc file needed for TJPCov
# inputs: ell binning set-up, n(z), n_eff, sigma_e, mask
# outputs for TJPCov: NaMaster workspace, sacc file containing n(z) and model spectra for all tomographic auto and cross-spectra
# the cosmic shear power spectra saved in this file IS NOT the mock spectra used in analysis, this is only used for making the covariance with TJPCov


import sys
import numpy as np
import pyccl as ccl
import sacc
import healpy as hp
import pymaster as nmt
import os


def generate_empty_sacc(nz_z_file, nz_realization_file, sacc_file, n_eff_tot):
    # mask_name = "data/lsst_binary_mask.fits"

    sigma_e = 0.26

    # set up, we consider two mocks
    ## y1-spec-z-lim, neff=9.78
    ## y10-deep, neff=24.4

    
    nz_data = np.load(nz_realization_file)['arr_0']
    nz_data = nz_data.mean(axis=0)
    z = np.load(nz_z_file)['arr_0']

    # number of tomographic bins
    num_z_bins = nz_data.shape[0]

    # set-up ell binning, use logarithmic bins
    ell_min = 20
    ell_max = 3000
    num_ell_bins = 20

    # set-up directories

    # read in mask
    # mask = hp.read_map(mask_name, verbose=False)
    # nside = hp.npix2nside(mask.size)
    nside = 2048

    n_eff = np.array([n_eff_tot / num_z_bins] * num_z_bins)  # n_eff per tomographic bin
    n_ell = sigma_e**2.0 / n_eff

    # set-up for sacc file and CCL
    s = sacc.Sacc()
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        n_s=0.95,
        sigma8=0.8,
        # transfer_function="bbks",
    )


    nz = np.zeros((num_z_bins, z.size))
    wl = []  # add tracer for ccl
    for i in range(num_z_bins):
        nz[i, :] = nz_data[i]
        s.add_tracer("NZ", "source_%s" % str(i), z, nz[i, :])
        wl.append(
            ccl.tracers.WeakLensingTracer(
                cosmo, [z, nz[i, :]], has_shear=True, ia_bias=None, use_A_ia=True
            )
        )
        tr = s.tracers["source_%s" % str(i)]
        tr.metadata["n_ell_coupled"] = 2e-9

    ell = np.arange(3 * nside, dtype="int32")
    cl = np.zeros((num_z_bins, num_z_bins, ell.size))
    CEE = sacc.standard_types.galaxy_shear_cl_ee

    # NaMaster friendly ell binning for TJPCov to work
    bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), num=num_ell_bins + 1)
    bpws = np.zeros(ell.size) - 1.0
    weights = np.ones(ell.size)
    w = np.zeros((ell.size, num_ell_bins))
    for nb in range(num_ell_bins):
        inbin = (ell <= bin_edges[nb + 1]) & (ell > bin_edges[nb])
        bpws[inbin] = nb
        norm_denom = np.sum(weights[inbin])
        weights[inbin] /= norm_denom
        w[inbin, nb] = 1.0
    b = nmt.NmtBin(nside, bpws=bpws, ells=ell, weights=weights, nlb=None)
    bin_ell = b.get_effective_ells()
    win = sacc.windows.BandpowerWindow(ell, w)

    # loop over tomographic bins bins
    for j in range(num_z_bins):
        for k in range(num_z_bins):
            print("j=", j, "k=", k)
            cl[j, k, :] = ccl.angular_cl(cosmo, wl[j], wl[k], ell)
            # arrays are shape (1,num_ell_bins) so need [0]
            cl_bin = b.bin_cell(np.array([cl[j, k, :]]))[0]
            
            nl_bin = b.bin_cell(np.array([np.ones(3 * nside) * n_ell[j]]))[0]


            for n in range(num_ell_bins):
                s.add_data_point(
                    CEE,
                    ("source_%s" % str(j), "source_%s" % str(k)),
                    cl_bin[n],
                    ell=bin_ell[n],
                    window=win,
                    i=j,
                    j=k,
                    n_ell=nl_bin[n],
                    window_ind=n,
                )
    s.save_fits(sacc_file, overwrite=True)
    return num_z_bins, n_eff


def main(nz_z_file, nz_realization_file, sacc_file, n_eff_tot, fsky):
    num_z_bins, n_eff = generate_empty_sacc(nz_z_file, nz_realization_file, sacc_file, n_eff_tot)
    print("N_eff = ", n_eff)
    from tjpcov.covariance_calculator import CovarianceCalculator

    config = {
        'tjpcov': {
            'sacc_file': sacc_file,
            'cosmo': 'data/fiducial_cosmology.yml',
            'cov_type': ['FourierGaussianFsky'],
            'IA': 0.0
        },
        'GaussianFsky': {'fsky': fsky}
    }

    for i in range(num_z_bins):
        config["tjpcov"][f"Ngal_source_{i}"] = n_eff[i]
        config["tjpcov"][f"sigma_e_source_{i}"] = 0.26

    cc = CovarianceCalculator(config)
    cov = cc.get_covariance()

    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        print("Futzing diagonal to get positive definite covariance")
        cov += cov.diagonal().min() * 1e-6 * np.eye(cov.shape[0])
        np.linalg.cholesky(cov)


    s = sacc.Sacc.load_fits(sacc_file)
    s.add_covariance(cov, overwrite=True)
    s.save_fits(sacc_file, overwrite=True)
