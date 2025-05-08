import numpy as np
import pickle as pkl
import os
import json
from glob import glob
import warnings
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from CPET.utils.io import save_numpy_as_dat, read_mat


class pca_pycpet:
    def __init__(self, options):
        """
        Initialize the PCA object with the following options:
        - pca_reload: boolean to reload a previously trained PCA object
        - save_pca: boolean to save the PCA object
        - inputpath: path to the input data
        - outputpath: path to save the PCA object and metadata json
        - verbose: boolean to print out the PCA explained variance

        Notes:
        - Default mode is to take an arbitrary file list and obtain principal components. To
        compare datasets and produce both individual and total PCs, change method from pca to
        pca_compare

        """

        # Note: need to figure out what this means:
        self.pca_reload = options["pca_reload"] if "pca_reload" in options else False
        self.save_pca_tf = options["save_pca"] if "save_pca" in options else False
        self.whitening = options["whitening"] if "whitening" in options else False
        self.verbose = options["verbose"] if "verbose" in options else True

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
                for i in self.file_list:
                    file_list.write(f"{i} \n")
        else:
            self.field_file_list = options["field_file_list"]

        self.load()

    def load(self):
        # go through the input path and load the data, read every .dat
        # file and store it in a numpy array

        self.data = []
        for file in self.field_file_list:
            self.data.append(read_mat(file))

        self.data = np.array(self.data)
        self.meta_data = read_mat(file, meta_data=True)

    def fit_and_transform(
        self,
    ):
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
        for i in range(0, len(self.pca_obj.explained_variance_ratio_)):
            if i == 0:
                cum_explained_var.append(self.pca_obj.explained_variance_ratio_[0])
            else:
                cum_explained_var.append(
                    self.pca_obj.explained_variance_ratio_[i] + cum_explained_var[i - 1]
                )

        self.cum_explained_var = cum_explained_var

        if self.verbose:
            print(
                "individual explained vars: \n"
                + str(self.pca_obj.explained_variance_ratio_)
            )
            print("cumulative explained vars ratio: \n" + str(cum_explained_var))

        if self.save_pca_tf:
            self.save_pca()

        return mat_transform, self.pca_obj

    # TODO: export PCA component to a .dat file

    def transform(self, data):
        mat_transform = data.reshape(
            data.shape[0], data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4]
        )

        mat_transform = self.pca_obj.transform(mat_transform)

        return mat_transform

    def save_pca(self):
        with open(self.outputpath + "pca.pkl", "wb") as f:
            pkl.dump(self.pca_obj, f)

    def load_pca(self):
        with open(self.outputpath + "pca.pkl", "rb") as f:
            self.pca_obj = pkl.load(f)

        with open(self.outputpath + "meta_data.pkl", "rb") as f:
            self.meta_data = json.load(f)

    def save_component(self, component=0, filename="pca_comp", shape=[21, 21, 21, 21]):
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
        Take as input a matrix that has been transformed by PCA and return the original matrix
        """
        mat = self.pca_obj.inverse_transform(mat)
        mat = mat.reshape(len(mat), shape[1], shape[2], shape[3], shape[4])
        return mat

    def plot_hist_along_component(
        x_list, pca_obj, label_list, components, loc, plot, type="png"
    ):
        if type(components) is list:
            fig, axs = plt.subplots(len(components), 1)
            names = ["PC {}".format(str(temp)) for temp in components]

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
                    "./pca_comps_alcdehydro/{}/pca_{}-{}.png".format(
                        loc, components[0], components[-1]
                    )
                )
            elif type == "svg":
                fig.savefig(
                    "./pca_comps_alcdehydro/{}/pca_{}-{}.svg".format(
                        loc, components[0], components[-1]
                    )
                )
            else:
                raise ValueError("Invalid plot format")
