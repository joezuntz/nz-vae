import numpy as np
from . import lae
import tensorflow as tf
import pickle

def generate_data(input_file, output_file):
    print(f"Loading data from {input_file} and {output_file}")
    
    # shape is (nrealizations, ntomo, nz)
    input_data = np.load(input_file)['arr_0']

    # normalise the input data
    mean_input = input_data.mean(axis=0)
    std_input = input_data.std(axis=0)
    std_input[std_input == 0] = 1
    input_data = (input_data - mean_input) / std_input
    print("Loaded input data")
    
    # output shape is (nrealizations, ndata)
    output_zip = np.load(output_file, allow_pickle=True)['arr_0'].item()
    print(output_zip.keys())
    output_data = output_zip['theory_vectors']

    # also normalise the output data
    mean_output = output_data.mean(axis=0)
    std_output = output_data.std(axis=0)
    std_output[std_output == 0] = 1
    output_data = (output_data - mean_output) / std_output
    print("Loaded output data")

    return input_data, output_data, (mean_input, std_input), (mean_output, std_output)



def main(nz_realization_file, data_vector_file, trained_model_file, history_file, latent_dim = 16, nepoch=1000, batch_size=200):
    input_data, output_data, _, _ = generate_data(input_file=nz_realization_file, output_file=data_vector_file)
    nbin = input_data.shape[1]
    nz = input_data.shape[2]
    ndata = output_data.shape[1]
    
    
    encoder, decoder = lae.make_conv_model(nbin, nz, ndata, latent_dim, verbose=True)
    model = lae.LAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim,nbin=nbin, nz=nz, ndata=ndata)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam())

    history = None
    try:
        history = model.fit(input_data, output_data, epochs=nepoch, batch_size=batch_size, verbose=True)
    except KeyboardInterrupt:
        pass

    model.save_lae(trained_model_file)

    if history is not None:
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)


if __name__ == "__main__":
    main(nepoch=1000)
    
