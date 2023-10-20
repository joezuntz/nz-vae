import numpy as np
from . import gen_training_dv
from . import gen_training_nz
from . import fit_model_lae
from . import gen_simulated_dv
from . import generate_empty_sacc

nz_realization_file = "data/nz_realization_array.npz"
nz_realization_z_file = "data/nz_realization_z.npz"
data_vector_file = "data/theory_vector_info.npz"
sacc_file = "data/raw_sacc.sacc"
alpha_dir = "data/alpha"
alpha_dir_hsc = "data/hsc_alphas"

trained_model_file = "data/model_{spec}.hdf5"
history_file = "data/history_{spec}.pkl"

training_delta_dv_plot = "data/training_delta_dv_{spec}.png"
lae_delta_dv_plot = "data/lae_delta_dv_{spec}.png"
chi2_plot = "data/chi2_{spec}.png"



def generate_training_nz_realizations(nsample, year):
    if year == "hsc":
        z, samples = gen_training_nz.generate_hsc(alpha_dir_hsc, nsample)
    else:
        z, samples = gen_training_nz.generate(
            year=year, alpha_dir=alpha_dir, nsample=nsample
        )
    np.savez(nz_realization_file, samples)
    np.savez(nz_realization_z_file, z)


def generate_sacc():
    generate_empty_sacc.main(nz_realization_z_file, nz_realization_file, sacc_file)

def generate_training_data_vectors(nsample):
    # should be run under MPI
    import mpi4py.MPI

    gen_training_dv.main(
        nz_realization_z_file,
        nz_realization_file,
        data_vector_file,
        nsample,
        mpi4py.MPI.COMM_WORLD,
    )


def fit_model(latent_dim, nepoch, batch_size, nsample):
    # Fit model mapping n(z) to latent space to data vectors
    spec = f"dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nsample}"
    fit_model_lae.main(
        nz_realization_file,
        data_vector_file,
        trained_model_file=trained_model_file.format(spec=spec),
        history_file=history_file.format(spec=spec),
        latent_dim=latent_dim,
        nepoch=nepoch,
        batch_size=batch_size,
    )


def generate_simulated_data_vectors(latent_dim, nepoch, batch_size, nsample):
    # Generate data vectors from latent space and compute their likelihoods as a test,
    # and visualize stuff
    spec = f"dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nsample}"
    model_file = trained_model_file.format(spec=spec)
    dv_plot = training_delta_dv_plot.format(spec=spec)
    lae_plot = lae_delta_dv_plot.format(spec=spec)
    c2_plot = chi2_plot.format(spec=spec)

    gen_simulated_dv.main(model_file, nz_realization_file, data_vector_file, dv_plot, lae_plot, c2_plot, latent_dim)


def run_mcmc():
    # run mcmc with marginalization over latent space
    pass



def main(stage):
    latent_dim = 16
    nepoch = 1000
    batch_size = 200
    nsample = 100_000
    year = "hsc"
    if stage == "gen_nz":
        generate_training_nz_realizations(nsample, year)
    elif stage == "gen_sacc":
        generate_sacc()
    elif stage == "gen_dv":
        generate_training_data_vectors(nsample)
    elif stage == "fit":
        fit_model(latent_dim, nepoch, batch_size, nsample)
    elif stage == "sim":
        generate_simulated_data_vectors(latent_dim, nepoch, batch_size, nsample)
    elif stage == "mcmc":
        run_mcmc()
    else:
        raise ValueError(f"Unknown stage {stage}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["gen_nz", "gen_sacc", "gen_dv", "fit", "sim", "mcmc"])
    args = parser.parse_args()
    main(args.stage)
