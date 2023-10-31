import numpy as np
from . import gen_training_dv
from . import gen_training_nz
from . import fit_model_lae
from . import gen_simulated_dv
from . import generate_empty_sacc
from . import mcmc

run = "y1"

nz_realization_file = f"data/{run}_nz_realization_array.npz"
nz_realization_z_file = f"data/{run}_nz_realization_z.npz"
data_vector_file = f"data/{run}_theory_vector_info.npz"
sacc_file = f"data/{run}_raw_sacc.sacc"
alpha_dir = "data/alpha"
alpha_dir_hsc = "data/hsc_alphas"

trained_model_file = "data/{run}_model_{spec}.hdf5"
history_file = "data/{run}_history_{spec}.pkl"

training_delta_dv_plot = "data/{run}_training_delta_dv_{spec}.png"
lae_delta_dv_plot = "data/{run}_lae_delta_dv_{spec}.png"
chi2_plot = "data/{run}_chi2_{spec}.png"

chain_file_fmt = "data/{run}_chain_{spec}.txt"
anim_data_file_fmt = "data/{run}_anim_data_{spec}.npy"


plot_file_fmt = "data/{run}_latent_param_plot_{spec}.pdf"
anim_file_fmt = "data/{run}_latent_param_anim_{spec}.gif"

nframe_per_dim = 11


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
    if run == "hsc":
        n_eff_tot = 9.78
        fsky = 0.01
    elif run == "y1":
        n_eff_tot = 10.0
        fsky = 12_300 / 41_253
    elif run == "y10":
        n_eff_tot = 27.0
        fsky = 14_300 / 41_253
    else:
        raise ValueError(f"Unknown run {run}")
    generate_empty_sacc.main(nz_realization_z_file, nz_realization_file, sacc_file, n_eff_tot, fsky)

def generate_training_data_vectors(nsample):
    # should be run under MPI
    import mpi4py.MPI

    gen_training_dv.main(
        nz_realization_z_file,
        nz_realization_file,
        data_vector_file,
        sacc_file,
        nsample,
        mpi4py.MPI.COMM_WORLD,
    )


def fit_model(model_name, latent_dim, nepoch, batch_size, nsample):
    # Fit model mapping n(z) to latent space to data vectors
    spec = f"{model_name}_dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nsample}"
    fit_model_lae.main(
        model_name,
        nz_realization_file,
        data_vector_file,
        trained_model_file=trained_model_file.format(spec=spec, run=run),
        history_file=history_file.format(spec=spec, run=run),
        latent_dim=latent_dim,
        nepoch=nepoch,
        batch_size=batch_size,
    )


def generate_simulated_data_vectors(model_name, latent_dim, nepoch, batch_size, nsample):
    # Generate data vectors from latent space and compute their likelihoods as a test,
    # and visualize stuff
    spec = f"{model_name}_dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nsample}"
    model_file = trained_model_file.format(spec=spec, run=run)
    dv_plot = training_delta_dv_plot.format(spec=spec, run=run)
    lae_plot = lae_delta_dv_plot.format(spec=spec, run=run)
    c2_plot = chi2_plot.format(spec=spec, run=run)

    gen_simulated_dv.main(model_name, model_file, nz_realization_file, data_vector_file, dv_plot, lae_plot, c2_plot)



def make_plot_data(model_name, latent_dim, nepoch, batch_size, nsample):
    spec = f"{model_name}_dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nsample}"
    model_file = trained_model_file.format(spec=spec, run=run)
    anim_data_file = anim_data_file_fmt.format(spec=spec, run=run)
    from mpi4py.MPI import COMM_WORLD
    comm = None if COMM_WORLD.size == 1  else COMM_WORLD
    mcmc.make_plot_data(model_name, model_file, nz_realization_file, data_vector_file, latent_dim, comm, anim_data_file, nframe_per_dim)

def make_plots(model_name, latent_dim, nepoch, batch_size, nsample):
    spec = f"{model_name}_dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nsample}"
    anim_data_file = anim_data_file_fmt.format(spec=spec, run=run)
    plot_file = plot_file_fmt.format(spec=spec, run=run)
    anim_file = anim_file_fmt.format(spec=spec, run=run)
    mcmc.make_static_plot(anim_data_file, latent_dim, nframe_per_dim, plot_file)
    mcmc.make_animated_plot(anim_data_file, latent_dim, nframe_per_dim, anim_file)


def run_mcmc(model_name, latent_dim, nepoch, batch_size, nsample):
    # run mcmc with marginalization over latent space
    spec = f"{model_name}_dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nsample}"
    model_file = trained_model_file.format(spec=spec, run=run)
    chain_file = chain_file_fmt.format(spec=spec, run=run)
    from mpi4py.MPI import COMM_WORLD
    comm = None if COMM_WORLD.size == 1  else COMM_WORLD

    mcmc.main(run, model_name, model_file, nz_realization_file, data_vector_file, latent_dim, comm, chain_file)



def main(stage, latent_dim, model_name, nepoch, batch_size):
    nsample = 100_000
    year = "1"
    if stage == "gen_nz":
        generate_training_nz_realizations(nsample, year)
    elif stage == "gen_sacc":
        generate_sacc()
    elif stage == "gen_dv":
        generate_training_data_vectors(nsample)
    elif stage == "fit":
        fit_model(model_name, latent_dim, nepoch, batch_size, nsample)
    elif stage == "sim":
        generate_simulated_data_vectors(model_name, latent_dim, nepoch, batch_size, nsample)
    elif stage == "mcmc":
        run_mcmc(model_name, latent_dim, nepoch, batch_size, nsample)
    elif stage == "plot_data":
        make_plot_data(model_name, latent_dim, nepoch, batch_size, nsample)
    elif stage == "plots":
        make_plots(model_name, latent_dim, nepoch, batch_size, nsample)
    else:
        raise ValueError(f"Unknown stage {stage}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["gen_nz", "gen_sacc", "gen_dv", "fit", "sim", "mcmc", "plot_data", "plots"])
    parser.add_argument("--latent-dim", type=int, default=12, help="Latent dimension size")
    parser.add_argument("--model-name", type=str, default='conv1', help="Name of model")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    args = parser.parse_args()
    main(args.stage, args.latent_dim, args.model_name, args.epochs, args.batch_size)
