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
    def __init__(self, options):
        # Logistics
        self.options = default_options_initializer(options)
        self.logger = logging.getLogger(__name__)  # Inherit logger from cpet.py
        self.m = self.options["CPET_method"]
        self.logger.info("Instantiating CPET, running method: {}".format(self.m))
        self.inputpath = self.options["inputpath"]
        self.outputpath = self.options["outputpath"]
        self.step_size = self.options["step_size"]
        if not os.path.exists(self.outputpath):
            print(
                "Output directory does not exist in current directory, creating: \n{}".format(
                    self.outputpath
                )
            )
            os.makedirs(self.outputpath)

        # Calculation-specific settings
        self.dimensions = self.options["dimensions"]

    def run(self):
        if self.m == "topo":
            self.run_topo()
        elif self.m == "topo_GPU":
            self.run_topo_GPU()
        elif self.m == "volume":
            self.run_volume()
        elif self.m == "volume_ESP":
            self.run_volume_ESP()
        elif self.m == "point_field":
            self.run_point_field()
        elif self.m == "point_mag":
            self.run_point_mag()
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
            print(
                "You have reached the limit of this package's capabilities at the moment, we do not support the function called as of yet"
            )
            exit()

    def run_topo(self, num=100000, benchmarking=False):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for i in range(num):
            if len(files_input) != 0:
                file = choice(files_input)
            else:
                print("No more files to process!")
                break
            files_input.remove(file)
            protein = file.split("/")[-1].split(".")[0]
            print("protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x.split(".")[-1] == "top"
            ]
            if protein + ".top" not in files_done:
                self.calculator = calculator(self.options, path_to_pdb=file)
                hist = self.calculator.compute_topo_complete_c_shared()
                if not benchmarking:
                    np.savetxt(self.outputpath + "/{}.top".format(protein), hist)
                if benchmarking:
                    np.savetxt(
                        self.outputpath
                        + "/{}_{}_{}_{}.top".format(
                            protein,
                            self.calculator.n_samples,
                            str(self.calculator.step_size)[2:],
                            self.replica,
                        ),
                        hist,
                    )
            else:
                print("Already done for protein: {}, skipping...".format(protein))

    def run_topo_GPU(self, num=100000, benchmarking=False):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for i in range(num):
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
                if not benchmarking:
                    np.savetxt(self.outputpath + "/{}.top".format(protein), hist)
                if benchmarking:
                    np.savetxt(
                        self.outputpath
                        + "/{}_{}_{}_{}.top".format(
                            protein,
                            self.calculator.n_samples,
                            str(self.calculator.step_size)[2:],
                            self.replica,
                        ),
                        hist,
                    )

    def run_volume(self, num=100000):
        """
        Get the electric fields along a grid of points in the box
        """

        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")

        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")

        for i in range(num):
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
                    "dimensions": self.dimensions,
                    "step_size": [self.step_size, self.step_size, self.step_size],
                    "num_steps": [mesh_shape[0], mesh_shape[1], mesh_shape[2]],
                    "transformation_matrix": self.calculator.transformation_matrix,
                    "center": self.calculator.center,
                }

                save_numpy_as_dat(
                    name=self.outputpath + "/{}_efield.dat".format(protein),
                    volume=field_box,
                    meta_data=meta_data,
                )

    def run_point_field(self):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        outfile = self.outputpath + "/point_field.dat"
        with open(outfile, "w") as f:
            for file in files_input:
                self.calculator = calculator(self.options, path_to_pdb=file)
                protein = file.split("/")[-1].split(".")[0]
                print("protein file: {}".format(protein))
                point_field = self.calculator.compute_point_field()
                f.write("{}:{}\n".format(protein, point_field))

    def run_point_mag(self):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        outfile = self.outputpath + "/point_mag.dat"
        with open(outfile, "w") as f:
            for file in files_input:
                self.calculator = calculator(self.options, path_to_pdb=file)
                protein = file.split("/")[-1].split(".")[0]
                print("protein file: {}".format(protein))
                point_field = self.calculator.compute_point_mag()
                f.write("{}:{}\n".format(protein, point_field))

    def run_volume_ESP(self, num=100000):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for i in range(num):
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
                    "dimensions": self.dimensions,
                    "step_size": [self.step_size, self.step_size, self.step_size],
                    "num_steps": [mesh_shape[0], mesh_shape[1], mesh_shape[2]],
                    "transformation_matrix": self.calculator.transformation_matrix,
                    "center": self.calculator.center,
                }
                save_numpy_as_dat(
                    name=self.outputpath + "/{}_esp.dat".format(protein),
                    volume=esp_box,
                    meta_data=meta_data,
                )

    def run_box_check(self, num=100000):
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
        print("Running the cluster analysis. Method type: {}".format(self.m))
        self.cluster = cluster(self.options)
        self.cluster.Cluster()

    def run_visualize_efield(self):
        print(
            "Visualizing the electric field. This module will load a ChimeraX session with the first protein and the electric field, and requires the electric field to be computed first."
        )
        files_input_pdb = glob(self.inputpath + "/*.pdb")
        if self.m == "visualize_field":
            files_input_efield = glob(self.inputpath + "/*_efield.dat")
        elif self.m == "visualize_esp":
            files_input_esp = glob(self.inputpath + "/*_esp.dat")
        if len(files_input_pdb) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input_pdb) > 1:
            warnings.warn(
                "More than one pdb file found in the input directory. Only the first will be visualized, .bild files will be generated for all of them though."
            )

        # Sort list of pdbs and efields
        files_input_pdb.sort()

        # Check to make sure each pdb file has a corresponding electric field file in the input path while visualizing fields
        for i in range(len(files_input_pdb)):
            if self.m == "visualize_field":
                # Modify efield file list to just have file name, not _efield.dat
                files_input_efield = [
                    efield.split("/")[-1].split("_efield")[0]
                    for efield in files_input_efield
                ]
                # Efield list is unsorted, so just check if the protein file is anywhere in the efield list
                if not any(
                    files_input_pdb[i].split("/")[-1].split(".")[0] in efield
                    for efield in files_input_efield
                ):
                    raise ValueError(
                        "No electric field file found for protein: {}".format(
                            files_input_pdb[i].split("/")[-1]
                        )
                    )
                print(
                    "Generating .bild file for the protein: {}".format(
                        files_input_pdb[i].split("/")[-1]
                    )
                )
                visualize.visualize_field(
                    path_to_pdb=files_input_pdb[i],
                    path_to_efield=self.inputpath
                    + "/"
                    + files_input_pdb[i].split("/")[-1].split(".")[0]
                    + "_efield.dat",
                    outputpath=self.outputpath,
                    options=self.options,
                )
            elif self.m == "visualize_esp":
                # Modify esp file list to just have file name, not _esp.dat
                files_input_esp = [
                    esp.split("/")[-1].split("_esp")[0] for esp in files_input_esp
                ]
                # Esp list is unsorted, so just check if the protein file is anywhere in the esp list
                if not any(
                    files_input_pdb[i].split("/")[-1].split(".")[0] in esp
                    for esp in files_input_esp
                ):
                    raise ValueError(
                        "No ESP file found for protein: {}".format(
                            files_input_pdb[i].split("/")[-1]
                        )
                    )
                print(
                    "Generating .bild file for the protein: {}".format(
                        files_input_pdb[i].split("/")[-1]
                    )
                )
                visualize.visualize_esp(
                    path_to_pdb=files_input_pdb[i],
                    path_to_esp=self.inputpath
                    + "/"
                    + files_input_pdb[i].split("/")[-1].split(".")[0]
                    + "_esp.dat",
                    outputpath=self.outputpath,
                    options=self.options,
                )
            # To-do: automatically visualize the electric field for the first protein, in dev mode for now

    def run_pca(self):
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
