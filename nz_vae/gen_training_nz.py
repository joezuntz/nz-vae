import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def draw_values(data, num=1, ret_mids=False):
    if num == 1:
        samp = np.random.dirichlet(data[:, 1])
        samp = samp / np.trapz(samp, data[:, 0])
    else:
        samp = np.random.dirichlet(data[:, 1], size=num)
        samp = np.array([el / np.trapz(el, data[:, 0]) for el in samp])

    if ret_mids == False:
        return samp
    else:
        return data[:, 0], samp


def generate(alpha_dir, year, nsample):
    alphas = []
    for i in range(5):
        alpha = np.loadtxt(f"{alpha_dir}/srd_y{year}_alpha_bin{i}.txt")
        alphas.append(alpha)

    z = alpha[:, 0]

    sample_array = np.array([draw_values(alphas[i], nsample) for i in range(5)])
    return z, sample_array

def generate_hsc(alpha_dir, nsample):
    alphas = []

    alpha_files = [
        "result_photometry_0_3_0_6_alpha.dat",
        "result_photometry_0_6_0_9_alpha.dat",
        "alpha_result_0_9_1_2.dat",
        "alpha_result_1_2_1_5.dat",
    ]
    alphas = [np.loadtxt(f"{alpha_dir}/{a}") for a in alpha_files]
    z = np.linspace(0.0, 3.000, 301)
    # z = (zbound[1:] + zbound[:-1]) / 2
    bin_width = z[1] - z[0]
    samples = [draw_values(a, nsample) for a in alphas]

    sample_interpolators = [interp1d(a[:, 0], s, fill_value = 0.0, kind = 'linear', bounds_error = False) for a,s in zip(alphas, samples)]

    nz_realization_array = np.array([f(z)[:] / np.sum(f(z)) / bin_width for f in sample_interpolators]).T
    nz_realization_array = np.swapaxes(nz_realization_array,0,1)
    nz_realization_array = np.swapaxes(nz_realization_array,1,2)
    print(z)
    nz_realization_array[:, :, 0] = 0.0
    return z, nz_realization_array