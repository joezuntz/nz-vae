import cosmosis
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation
import os 

def setup_parameters(model_name, model_file, nz_realization_file, data_vector_file, latent_dim):
    ini = cosmosis.Inifile("data/params.ini")
    sec = "lae_sacc_like"
    ini[sec, "model_file"] = model_file
    ini[sec, "model_name"] = model_name
    ini[sec, "nz_realization_file"] = nz_realization_file
    ini[sec, "data_vector_file"] = data_vector_file

    nz = "compressed_nz_parameters"
    values = cosmosis.Inifile("data/values.ini")
    priors = cosmosis.Inifile(None)
    values.add_section(nz)
    priors.add_section(nz)
    for i in range(latent_dim):
        values[nz, f"theta_{i}"] = "-3.0  0.0  3.0"
        priors[nz, f"theta_{i}"] = "gaussian 0.0 1.0"
    return ini, values, priors

def main(run, model_name, model_file, nz_realization_file, data_vector_file, latent_dim, comm, chain_file):
    os.environ["RUN_NAME"] = run
    ini, values, priors = setup_parameters(model_name, model_file, nz_realization_file, data_vector_file, latent_dim)
    ini["runtime", "sampler"] = "emcee"
    ini["output", "filename"] = chain_file
    ini.remove_option("lae_sacc_like", "model_file")
    values.remove_section("compressed_nz_parameters")

    if comm is None:
        return cosmosis.run_cosmosis(ini, values=values, priors=priors)
    else:
        with cosmosis.MPIPool(comm=comm) as pool:
            return cosmosis.run_cosmosis(ini, pool=pool)

def make_plot_data(model_name, model_file, nz_realization_file, data_vector_file, latent_dim, comm, anim_data_file, nframe_per_dim = 11):
    ini, values, priors = setup_parameters(model_name, model_file, nz_realization_file, data_vector_file, latent_dim)
    pipeline = cosmosis.LikelihoodPipeline(ini, values=values, priors=priors)
    
    nv = pipeline.nvaried
    rank = 0 if comm is None else comm.rank
    size = 1 if comm is None else comm.size
    x = pipeline.start_vector()
    block = pipeline.run_parameters(x)
    baseline_data_vector = block['data_vector', '2pt_theory']    
    output_size = baseline_data_vector.size
    output = np.zeros((latent_dim, nframe_per_dim, output_size))

    p = 0
    for i in range(latent_dim):
        x = pipeline.start_vector()
        j = nv - latent_dim + i
        for k, theta in enumerate(np.linspace(-2.0, 2.0, nframe_per_dim)):
            if p % size == rank:
                x[j] = theta
                print(rank, x)
                sys.stdout.flush()
                block = pipeline.run_parameters(x)
                output[i, k, :] = block["data_vector", "2pt_theory"]
            p += 1
    if comm is not None:
        output = comm.allreduce(output)
    if rank == 0:
        anim_data = {'data': output, 'baseline': baseline_data_vector}
        np.save(anim_data_file, anim_data)


def make_static_plot(anim_data_file, latent_dim, nframe_per_dim, plot_file, scaling=1e12):
    anim_data = np.load(anim_data_file, allow_pickle=True).item()
    baseline_data_vector = anim_data['baseline']
    data = anim_data['data']

    font = {'family' : 'serif',
            'size'   : 12}

    plt.rc('font', **font)
    nrow = int(np.ceil(latent_dim/2))
    fig, axes = plt.subplots(nrow, 2, figsize=(nrow * 2, 6))

    for d in range(latent_dim):
        ax = axes[d % nrow, d // nrow]
        if d % nrow == nrow - 1:
            ax.set_xlabel("$C_\ell$ Index")
        else:
            ax.set_xticks([])
        if d // nrow == 0:
            ax.set_ylabel("$10^{12}\Delta C_\ell$")
        for i in range(nframe_per_dim):
            color = plt.cm.viridis(i/nframe_per_dim)
            y = data[d, i] - baseline_data_vector
            ax.plot(y * scaling, color=color)
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(plot_file)
    plt.close()


def make_animated_plot(anim_data_file, latent_dim, nframe_per_dim, anim_file, scaling=1e12):
    anim_data = np.load(anim_data_file, allow_pickle=True).item()
    baseline_data_vector = anim_data['baseline']
    data = anim_data['data']

    fig = plt.figure(figsize=(6, int(1.5 * latent_dim)), constrained_layout=True)
    gs = fig.add_gridspec(ncols=4, nrows=latent_dim)
    x1 = np.linspace(-3, 3, 100)
    y1 = np.exp(-0.5 * x1**2)
    x2 = np.linspace(-2, 2, nframe_per_dim)
    y2 = np.exp(-0.5 * x2**2)
    dots = []
    lines = []
    for i in range(latent_dim):
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(x1, y1)
        dot1, = ax1.plot(x2[0], y2[0], 'r.', markersize=10)
        ax2 = fig.add_subplot(gs[i, 1:])
        y = data[i, 0] - baseline_data_vector
        ymax = (data[i].max(0) - baseline_data_vector).max()
        ymin = (data[i].min(0) - baseline_data_vector).min()
        print(ymin)
        line1, = ax2.plot(y * scaling)
        ax2.set_ylim(ymin * scaling, ymax* scaling)
        
        ax1.set_yticks([])
        if i != latent_dim - 1:
            ax1.set_xticks([])
            ax2.set_xticks([])
        
        lines.append(line1)
        dots.append(dot1)

        
    def update(i):
        for j in range(latent_dim):
            dots[j].set_xdata([x2[i]])
            dots[j].set_ydata([y2[i]])
            ynew = (data[j, i] - baseline_data_vector) * scaling
            lines[j].set_ydata(ynew)

            
    anim = matplotlib.animation.FuncAnimation(fig, update, frames=11)
    anim.save(anim_file)
    plt.close()
