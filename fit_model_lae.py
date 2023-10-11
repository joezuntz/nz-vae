import numpy as np
import nz_vae.lae as lae
import tensorflow as tf


def generate_data(input_file = "data/nz_realization_array.npy", output_file="data/dirichlet_theory_vectors.txt"):
    # input data is normalized n(z)
    # output data is normalized data vector

    # shape is (nrealizations, ntomo, nz)
    input_data = np.load(input_file)

    # normalise the input data
    mean_input = input_data.mean(axis=0)
    std_input = input_data.std(axis=0)
    input_data = (input_data - mean_input) / std_input

    # output shape is (nrealizations, ndata)
    output_data = np.loadtxt()

    # also normalise the output data
    mean_output = output_data.mean(axis=0)
    std_output = output_data.std(axis=0)
    output_data = (output_data - mean_output) / std_output

    return input_data, output_data, (mean_input, std_input), (mean_output, std_output)



def main(nepoch=10, batch_size=200):
    input_data, output_data, _, _ = generate_data()
    nreal = input_data.shape[0]
    nbin = input_data.shape[1]
    nz = input_data.shape[2]
    ndata = output_data.shape[1]
    latent_dim = 16
    encoder, decoder = lae.make_conv_model(nbin, nz, ndata, latent_dim, verbose=True)
    lae = lae.LAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim,)
    lae.compile(optimizer=tf.keras.optimizers.legacy.Adam())
    history = lae1.fit(input_data, output_data, nepoch=nepoch, batch_size=batch_size, verbose=True)
    lae.save(f"lae_model_dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nreal}.keras")


if __name__ == "__main__":
    main()
    