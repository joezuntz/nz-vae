from . import fit_model_lae
from . import lae
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def main(
    model_file,
    nz_realization_file,
    data_vector_file,
    training_delta_dv_plot,
    lae_delta_dv_plot,
    chi2_plot,
):
    (
        _,
        training_dv,
        _,
        (dv_mean, dv_std),
    ) = fit_model_lae.generate_data(nz_realization_file, data_vector_file)


    model = lae.LAE.load_lae(model_file)
    

    n = 100_000
    z_sample = np.random.normal(size=(n, model.latent_dim))
    dv_generated_normalized = model.decoder.predict(z_sample)
    dv_generated = dv_generated_normalized * dv_std + dv_mean

    plt.figure()
    for i in range(10):
        plt.plot(training_dv[i], "--")
    plt.savefig(training_delta_dv_plot)
    plt.close()

    plt.figure()
    for i in range(10):
        plt.plot(dv_generated_normalized[i])
    plt.savefig(lae_delta_dv_plot)
    plt.close()

    dv_data = np.load(data_vector_file, allow_pickle=True)["arr_0"].item()
    original_chi2 = dv_data["chi2"]
    y0 = dv_data["y0"]
    P = dv_data["precision_matrix"]
    d = dv_generated - y0
    generated_chi2 = np.einsum("ki,ij,kj->k", d, P, d)

    plt.hist(original_chi2, bins=100, histtype='step', label="original", density=1, range=(0, 5))
    plt.hist(generated_chi2, bins=100, histtype='step', label="generated", density=1, range=(0, 5))
    plt.legend()
    plt.savefig(chi2_plot)
    plt.close()
