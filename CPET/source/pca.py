import numpy as np
import pickle as pkl
from glob import glob
import warnings
import logging
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from CPET.utils.io import save_numpy_as_dat, read_mat, default_options_initializer

log = logging.getLogger(__name__)


class pca_pycpet:
    """
    Principal component analysis of a set of electric field volumes.

    Creating the object reads the field files (every ``.dat`` file in
    ``inputpath``, or ``field_file_list`` if given) and, with ``pca_reload``,
    loads a fitted PCA from ``outputpath/pca.pkl``. The file list is written to
    ``outputpath/field_file_list.txt``. Call :meth:`fit_and_transform` to run the
    PCA.

    Parameters
    ----------
    options : dict
        Options. Reads ``inputpath``, ``outputpath`` and the PCA options
        (``n_pca_components``, ``field_file_list``, ``pca_reload``, ``save_pca``,
        ``whitening``, ``verbose``); see the Options Reference.
    """
    def __init__(self, options):
        """
        Read the field files and, with ``pca_reload``, load a fitted PCA.
        """
        # Fill defaults in case pca_pycpet is used outside of CPET.py
        options = default_options_initializer(options)

        # Note: need to figure out what this means:
        self.pca_reload = options["pca_reload"]
        self.save_pca_tf = options["save_pca"]
        self.whitening = options["whitening"]
        self.verbose = options["verbose"]

        self.inputpath = options["inputpath"]
        self.outputpath = options["outputpath"]
        if "n_pca_components" not in options:
            warnings.warn(
                "No number of PCA components specified, using default value of 10"
            )
        self.components = (
            options["n_pca_components"] if "n_pca_components" in options else 10
        )

        if self.pca_reload:
            self.load_pca()
        else:
            self.pca_obj = None

        # user should also be able to specify list of files alternatively
        if "field_file_list" not in options:
            # load dataset and metadata
            self.field_file_list = []
            for file in glob(self.inputpath + "/*.dat"):
                self.field_file_list.append(file)
            if len(self.field_file_list) == 0:
                raise ValueError("No data found in the input path!")
            field_file_name = self.outputpath + "/field_file_list.txt"
            with open(field_file_name, "w") as file_list:
                for i in self.field_file_list:
                    file_list.write(f"{i} \n")
        else:
            self.field_file_list = options["field_file_list"]

        self.load()

    def load(self):
        # go through the input path and load the data, read every .dat
        # file and store it in a numpy array

        """
        Read every file in ``field_file_list`` into ``data``, and the last file's header into ``meta_data``.
        """
        self.data = []
        for file in self.field_file_list:
            self.data.append(read_mat(file))

        self.data = np.array(self.data)
        self.meta_data = read_mat(file, meta_data=True)

    def fit_and_transform(self,):
        """
        Fit the PCA, or use the loaded one, and project the fields onto the principal components.

        With ``verbose``, logs the explained variance ratios. With ``save_pca``,
        saves the PCA to ``outputpath/pca.pkl``. The projections are returned, not
        written to a file.

        Returns
        -------
        mat_transform : np.ndarray
            Projection of each field onto the components, shape
            (n_files, n_pca_components).
        pca_obj : sklearn.decomposition.PCA
            The fitted PCA.
        """
        mat_transform = self.data.reshape(
            self.data.shape[0],
            self.data.shape[1]
            * self.data.shape[2]
            * self.data.shape[3]
            * self.data.shape[4],
        )

        if self.pca_obj == None:
            self.pca_obj = PCA(n_components=self.components, whiten=self.whitening)
            mat_transform = self.pca_obj.fit_transform(mat_transform)

        else:
            mat_transform = self.pca_obj.transform(mat_transform)

        cum_explained_var = []
        for i in range(len(self.pca_obj.explained_variance_ratio_)):
            if i == 0:
                cum_explained_var.append(self.pca_obj.explained_variance_ratio_[0])
            else:
                cum_explained_var.append(
                    self.pca_obj.explained_variance_ratio_[i] + cum_explained_var[i - 1]
                )

        self.cum_explained_var = cum_explained_var

        if self.verbose:
            log.info(
                f"Individual explained variance ratios: {self.pca_obj.explained_variance_ratio_}"
            )
            log.info(f"Cumulative explained variance ratios: {cum_explained_var}")

        if self.save_pca_tf:
            self.save_pca()

        return mat_transform, self.pca_obj

    # TODO: export PCA component to a .dat file

    def transform(self, data):
        """
        Project field volumes onto the fitted principal components.

        Parameters
        ----------
        data : np.ndarray
            Field volumes, shape (n_fields, nx, ny, nz, 3).

        Returns
        -------
        np.ndarray
            Projections, shape (n_fields, n_pca_components).
        """
        mat_transform = data.reshape(
            data.shape[0], data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4]
        )

        mat_transform = self.pca_obj.transform(mat_transform)

        return mat_transform

    def save_pca(self):
        """
        Save the fitted PCA to ``outputpath/pca.pkl``.
        """
        with open(self.outputpath + "/pca.pkl", "wb") as f:
            pkl.dump(self.pca_obj, f)

    def load_pca(self):
        # meta_data is set from the input files by load(), which runs after this
        """
        Load a fitted PCA from ``outputpath/pca.pkl``.
        """
        with open(self.outputpath + "/pca.pkl", "rb") as f:
            self.pca_obj = pkl.load(f)

    def save_component(self, component=0, filename="pca_comp", shape=[21, 21, 21, 21]):
        """
        Save principal components as field files.

        Notes
        -----
        Not working: it reads a ``"shape"`` key that ``meta_data`` doesn't have and
        calls ``save_numpy_as_dat`` with the wrong arguments.
        """
        if type(component) == list:
            for i, comp in enumerate(component):
                component_field = self.pca_obj.components_[component].reshape(
                    self.meta_data["shape"][0],
                    self.meta_data["shape"][1],
                    self.meta_data["shape"][2],
                    self.meta_data["shape"][3],
                )
                save_numpy_as_dat(comp, filename + "_comp_" + str(comp))
        else:
            component_field = (
                self.pca_obj.components_[component].reshape(
                    self.meta_data["shape"][0],
                    self.meta_data["shape"][1],
                    self.meta_data["shape"][2],
                    self.meta_data["shape"][3],
                ),
            )

            save_numpy_as_dat(component_field, filename)

    def unwrap_pca(self, mat, shape):
        """
        Map PCA projections back to field volumes.

        Parameters
        ----------
        mat : np.ndarray
            Projections, shape (n_fields, n_pca_components).
        shape : sequence of int
            Shape of the original data, (n_fields, nx, ny, nz, 3); only the last four
            entries are used.

        Returns
        -------
        np.ndarray
            Reconstructed field volumes, shape (n_fields, nx, ny, nz, 3).
        """
        mat = self.pca_obj.inverse_transform(mat)
        mat = mat.reshape(len(mat), shape[1], shape[2], shape[3], shape[4])
        return mat

    def plot_hist_along_component(
        x_list, pca_obj, label_list, components, loc, plot, type="png"
    ):
        """
        Plot the distribution of each dataset along chosen principal components.

        Notes
        -----
        Not working: it has no ``self`` argument, its ``type`` argument hides the
        built-in ``type()`` it calls, it uses functions that aren't defined, and it
        saves to a hard-coded folder.
        """
        if type(components) is list:
            fig, axs = plt.subplots(len(components), 1)
            names = [f"PC {str(temp)}" for temp in components]

            for ind, comp in enumerate(components):
                list_data = []
                for _, x in enumerate(x_list):
                    x_pca, _ = pca(x, pca=pca_obj, verbose=False)
                    x_pca = x_pca[:, comp]
                    list_data.append(x_pca)

                xs_list, density_list = create_hist_plot(list_data, kernel=0.25)

                for i in range(len(xs_list)):
                    axs[ind].plot(
                        xs_list[i],
                        density_list[i],
                        color=[
                            "#1f77b4",
                            "#ff7f0e",
                            "#2ca02c",
                            "#d62728",
                            "#9467bd",
                            "#8c564b",
                            "#e377c2",
                        ][i],
                        label=label_list[i],
                    )

                axs[ind].set_title(names[ind], fontsize=16)
                axs[ind].set_ylabel("Density", fontsize=14)
                axs[ind].legend()
                for axs[ind] in axs:
                    axs[ind].tick_params(labelsize=14)

            fig.set_size_inches(8, 25)
            # fig.show()
            if plot:
                fig.show()
            if type == "png":
                fig.savefig(
                    f"./pca_comps_alcdehydro/{loc}/pca_{components[0]}-{components[-1]}.png"
                )
            elif type == "svg":
                fig.savefig(
                    f"./pca_comps_alcdehydro/{loc}/pca_{components[0]}-{components[-1]}.svg"
                )
            else:
                raise ValueError("Invalid plot format")
