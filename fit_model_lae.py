import numpy as np
import nz_vae.lae as lae
import tensorflow as tf
import pickle

def generate_data(input_file = "data/nz_realization_array.npy", output_file="data/dirichlet_theory_vectors.npy"):
    print(f"Loading data from {input_file} and {output_file}")
    
    # shape is (nrealizations, ntomo, nz)
    input_data = np.load(input_file)

    # normalise the input data
    mean_input = input_data.mean(axis=0)
    std_input = input_data.std(axis=0)
    std_input[std_input == 0] = 1
    input_data = (input_data - mean_input) / std_input
    print("Loaded input data")
    
    # output shape is (nrealizations, ndata)
    output_data = np.load(output_file)

    # also normalise the output data
    mean_output = output_data.mean(axis=0)
    std_output = output_data.std(axis=0)
    std_output[std_output == 0] = 1
    output_data = (output_data - mean_output) / std_output
    print("Loaded output data")

    return input_data, output_data, (mean_input, std_input), (mean_output, std_output)



def main(nepoch=1000, batch_size=200):
    input_data, output_data, _, _ = generate_data()
    nreal = input_data.shape[0]
    nbin = input_data.shape[1]
    nz = input_data.shape[2]
    ndata = output_data.shape[1]
    latent_dim = 16
    
    encoder, decoder = lae.make_conv_model(nbin, nz, ndata, latent_dim, verbose=True)
    model = lae.LAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim,)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam())

    history = None
    try:
        history = model.fit(input_data, output_data, epochs=nepoch, batch_size=batch_size, verbose=True)
    except KeyboardInterrupt:
        pass

    model.save_weights(f"lae_weights_dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nreal}.hdf5")

    if history is not None:
        with open(f'history_dim{latent_dim}_eps{nepoch}_bat{batch_size}_real{nreal}.pkl', 'wb') as f:
            pickle.dump(history.history, f)


if __name__ == "__main__":
    main(nepoch=1000)
    
