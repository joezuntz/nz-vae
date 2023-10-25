from cosmosis.datablock import option_section
import numpy as np
from nz_vae import fit_model_lae
from nz_vae import lae
import os
import sys

csl_dir = os.environ["CSL_DIR"]
sacc_like_path = os.path.join(csl_dir, "likelihood", "sacc")
sys.path.append(sacc_like_path)
import sacc_like

class LAESaccLike(sacc_like.SaccClLikelihood):
    def __init__(self, options):
        super().__init__(options)
        self.dv_mean, self.dv_std, self.lae_model = self.setup_lae(options)

    def setup_lae(self, options):

        if options.has_value("model_file"):
            model_file = options["model_file"]
            model_name = options["model_name"]
            nz_file = options["nz_realization_file"]
            dv_file = options["data_vector_file"]

            _, _, _, (dv_mean, dv_std) = fit_model_lae.generate_data(nz_file, dv_file)

            model = lae.LAE.load_lae(model_name, model_file)
        else:
            return None, None, None

        return dv_mean, dv_std, model

    def contaminate_theory_vector(self, block, theory, dataset_name, angle, bin1, bin2):
        if self.lae_model is None:
            return
        # Generate the latent parameters, which should be
        # normally distributed
        latent_dim = self.lae_model.latent_dim
        section = "compressed_nz_parameters"
        # This needs to be a 1 x latent_dim array for the predict method to be happy.
        q = [[block[section, f"theta_{i}"] for i in range(latent_dim)]]
        q = np.array(q)

        normalized_delta = self.lae_model.decoder.predict(q, verbose=0)[0]
        delta = normalized_delta * self.dv_std

        block["data_vector_contaminant", "delta"] = delta
        theory += delta

setup, execute, cleanup = LAESaccLike.build_module()
