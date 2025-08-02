from CPET.source.calculator import calculator
from CPET.source.cluster import cluster
from CPET.source.pca import pca_pycpet

import CPET.utils.visualize as visualize
from CPET.utils.io import save_numpy_as_dat, default_options_initializer
from CPET.utils.calculator import report_inside_box

from glob import glob
from random import choice
import os
import numpy as np
import warnings
import logging


class CPET:
    """The main class for the CPET package. This class is responsible for running almost any calculation, either through cpet.py or other scripts.

    Parameters
    ----------
    options : dict
        A dictionary containing the options for the CPET package. An empty dictionary can be passed, in which the default options initializer will fully override the options with the default options. The options can also be passed as a JSON file, which will be loaded and used to initialize the CPET package.

    Attributes
    ----------
    self.options : dict
        A dictionary containing the options for the CPET package.
    self.logger : logging.Logger
        A logger for the CPET package, inheriting from the logger in cpet.py. Logging is not guaranteed when using an auxiliary script
    self.m : str
        The method to be used for the CPET package, set from the options.
    self.inputpath : str
        The input path for CPET, set from the options.
    self.outputpath : str
        The output path for CPET, set from the options.
    """

    def __init__(self, options):
        # Logistics
        self.options = default_options_initializer(options)
        self.logger = logging.getLogger(__name__)  # Inherit logger from cpet.py
        self.m = self.options["CPET_method"]
        print("Instantiating CPET, method: {}".format(self.m))
        self.inputpath = self.options["inputpath"]
        self.outputpath = self.options["outputpath"]
        if not os.path.exists(self.outputpath):
            print(
                "Output directory does not exist in current directory, creating: \n{}".format(
                    self.outputpath
                )
            )
            os.makedirs(self.outputpath)

    def run(self):
        """Run the CPET package based on the method specified in the options. This method will call the appropriate method based on the value of self.m."""
        if self.m == "topo":
            self.run_topo()
        elif self.m == "topo_GPU":
            self.run_topo(gpu=True)
        elif self.m == "volume":
            self.run_volume()
        elif self.m == "volume_ESP":
            self.run_volume_ESP()
        elif self.m == "point_field" or self.m == "point_mag":
            self.run_point()
        elif (
            self.m == "cluster"
            or self.m == "cluster_volume"
            or self.m == "cluster_volume_tensor"
            or self.m == "cluster_volume_esp_tensor"
        ):
            self.run_cluster()
        elif self.m == "box_check":
            self.run_box_check()
        elif self.m == "visualize_field" or self.m == "visualize_esp":
            self.run_visualize_efield()
        elif self.m == "pca" or self.m == "pca_compare":
            self.run_pca()
        else:
            ValueError(
                "Method {} not recognized. Please check the options file or the command line arguments.".format(
                    self.m
                )
            )

    def run_topo(self, gpu=False):
        """Run the electric field topology calculation for a number of proteins in the input directory.
        Picks a pdb file at random from the input directory and runs the topology calculation on it, allowing for dirty parallel runs of this function.

        Parameters
        ----------
        gpu : bool, optional
            If True, runs the topology calculation on the GPU. Default is False, which runs the calculation on the CPU.
        """
        if gpu:
            runtype = "compute_topo_GPU_batch_filter"
        else:
            runtype = "compute_topo_complete_c_shared"
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            logging.warning("Only one pdb file found in the input directory")
        for i in range(len(files_input) * 100):
            if len(files_input) != 0:
                file = choice(files_input)
            else:
                print("No more files to process!")
                break
            files_input.remove(file)
            protein = file.split("/")[-1].split(".")[0]
            logging.info("Protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x.split(".")[-1] == "top"
            ]
            if protein + ".top" not in files_done:
                self.calculator = calculator(self.options, path_to_pdb=file)
                hist = getattr(self.calculator, runtype)()
                np.savetxt(self.outputpath + "/{}.top".format(protein), hist)
            else:
                print("Already done for protein: {}, skipping...".format(protein))

    '''
    Getting rid of this function, as it is already covered by above run_topo
    def run_topo_GPU(self):
        """Run the electric field topology calculation (GPU-accelerated) for a number of proteins in the input directory.
        Picks a pdb file at random from the input directory and runs the topology calculation on it, allowing for dirty parallel runs of this function.

        """
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for i in range(len(files_input)*100):
            if len(files_input) != 0:
                file = choice(files_input)
            else:
                break
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            files_input.remove(file)
            print("protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x.split(".")[-1] == "top"
            ]
            if protein + ".top" not in files_done:
                hist = self.calculator.compute_topo_GPU_batch_filter()
                np.savetxt(self.outputpath + "/{}.top".format(protein), hist)
    '''

    def run_volume(self):
        """Get the electric fields along a grid of points in the box"""

        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")

        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")

        for i in range(len(files_input) * 100):
            if len(files_input) != 0:
                file = choice(files_input)
            else:
                print("No more files to process!")
                break
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            files_input.remove(file)
            print("protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x[-11:] == "_efield.dat"
            ]

            if protein + "_efield.dat" not in files_done:
                field_box, mesh_shape = self.calculator.compute_box()
                print(field_box.shape)
                meta_data = {
                    "dimensions": self.calculator.dimensions,
                    "step_size": [
                        self.calculator.step_size,
                        self.calculator.step_size,
                        self.calculator.step_size,
                    ],
                    "num_steps": [mesh_shape[0], mesh_shape[1], mesh_shape[2]],
                    "transformation_matrix": self.calculator.transformation_matrix,
                    "center": self.calculator.center,
                }

                save_numpy_as_dat(
                    name=self.outputpath + "/{}_efield.dat".format(protein),
                    volume=field_box,
                    meta_data=meta_data,
                )

    def run_point(self):
        """Get the electric field/magnitude at the center of the box"""
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        if self.m == "point_field":
            outfile = self.outputpath + "/point_field.dat"
            runtype = "compute_point_field"
        elif self.m == "point_mag":
            outfile = self.outputpath + "/point_mag.dat"
            runtype = "compute_point_mag"
        with open(outfile, "w") as f:
            for file in files_input:
                self.calculator = calculator(self.options, path_to_pdb=file)
                protein = file.split("/")[-1].split(".")[0]
                print("protein file: {}".format(protein))
                point_field_or_mag = getattr(self.calculator, runtype)()
                # Save the point field or magnitude to the output file
                f.write("{}:{}\n".format(protein, point_field_or_mag))

    def run_volume_ESP(self):
        """Get the electrostatic potential along a grid of points in the box"""
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for i in range(len(files_input) * 100):
            if len(files_input) != 0:
                file = choice(files_input)
            else:
                print("No more files to process!")
                break
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            files_input.remove(file)
            print("protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x[-11:] == "_esp.dat"
            ]
            if protein + "_esp.dat" not in files_done:
                esp_box, mesh_shape = self.calculator.compute_box_ESP()
                print(esp_box.shape)
                meta_data = {
                    "dimensions": self.calculator.dimensions,
                    "step_size": [
                        self.calculator.step_size,
                        self.calculator.step_size,
                        self.calculator.step_size,
                    ],
                    "num_steps": [mesh_shape[0], mesh_shape[1], mesh_shape[2]],
                    "transformation_matrix": self.calculator.transformation_matrix,
                    "center": self.calculator.center,
                }
                save_numpy_as_dat(
                    name=self.outputpath + "/{}_esp.dat".format(protein),
                    volume=esp_box,
                    meta_data=meta_data,
                )

    def run_box_check(self):
        """Reports atoms that are inside the user-defined box, to check for potential e-field conflicts"""
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for file in files_input:
            if "filter_radius" in self.options or "filter_resnum" in self.options:
                # Error out, radius not compatible
                raise ValueError(
                    "filter_radius/filter_resnum is not compatible with box_check. Please remove from options"
                )
            # Need to not filter in box to check, but can filter all else
            self.options["filter_in_box"] = False
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            print("protein file: {}".format(protein))
            report_inside_box(self.calculator)
        print("No more files to process!")

    def run_cluster(self):
        """Run clustering (for a variety of types of field/esp data)"""
        print("Running the cluster analysis. Method type: {}".format(self.m))
        self.cluster = cluster(self.options)
        self.cluster.Cluster()

    def run_visualize_efield(self):
        """Visualize the electric field/electrostatic potential file for ChimeraX"""
        print(
            "Visualizing the electric field or electrostatic potential for use in ChimeraX"
        )
        if self.m == "visualize_field":
            files_input = glob(self.inputpath + "/*_efield.dat")
        elif self.m == "visualize_esp":
            files_input = glob(self.inputpath + "/*_esp.dat")

        # Check to make sure each pdb file has a corresponding electric field file in the input path while visualizing fields
        for file in files_input:
            if self.m == "visualize_field":
                visualize.visualize_field(
                    path_to_efield=file,
                    outputpath=self.outputpath,
                    options=self.options,
                )
            elif self.m == "visualize_esp":
                visualize.visualize_esp(
                    path_to_esp=file,
                    outputpath=self.outputpath,
                    options=self.options,
                )

    def run_pca(self):
        """Run PCA on a set of electric fields or groups of electric fields (good for mutation comparison)"""
        if self.m == "pca":
            self.pca = pca_pycpet(self.options)
            self.pca.fit_and_transform()
        elif self.m == "pca_compare":
            # Check for provided directories list for comparison
            if "inputpath_list" not in self.options:
                raise ValueError(
                    "No inputpath_list provided for PCA comparison mode. Please provide a list of directories that contain field files in the output file, or use the 'pca' method instead."
                )
            if "outputpath_list" not in self.options:
                warnings.warn(
                    "No outputpath_list provided. Using default outputpath_list based on inputpath_list"
                )
                # Add 'pca_out' to the end of each input path
                self.options["outputpath_list"] = [
                    path + "/pca_out" for path in self.options["inputpath_list"]
                ]
            if self.options["pca_combined_only"] == False:
                # Run PCA for each individual variant
                for inputpath, outputpath in zip(
                    self.options["inputpath_list"], self.options["outputpath_list"]
                ):
                    self.options["inputpath"] = inputpath
                    self.options["outputpath"] = outputpath
                    print(
                        "Running PCA for variant: {}".format(inputpath.split("/")[-1])
                    )
                    self.pca = pca_pycpet(self.options)
                    self.pca.fit_and_transform()
            else:
                from CPET.utils.io import pull_mats_from_MD_folder

                # Pull all field files from all variants
                all_field_files = []
                for i in range(len(self.options["inputpath_list"])):
                    all_field_files.extend(
                        pull_mats_from_MD_folder(self.options["inputpath_list"][i])
                    )
                all_fields = np.concatenate(all_field_files, axis=0)

                # Make a directory called 'pca_combined' in the current directory
                if not os.path.exists("pca_combined"):
                    os.makedirs("pca_combined")
                self.options["outputpath"] = "./pca_combined"
            # PCA for combined set of variants
            # TBD
