from CPET.source.calculator import calculator
from CPET.source.cluster import cluster
from CPET.source.pca import pca_pycpet

from CPET.utils import visualize
from CPET.utils.io import (
    save_numpy_as_dat,
    default_options_initializer,
    get_input_files,
    check_valid_options,
)
from CPET.utils.calculator import report_inside_box

from glob import glob
from random import choice
import os
import numpy as np
import logging

log = logging.getLogger(__name__)


class CPET:
    """
    Run any PyCPET method from an options dictionary.

    ``cpet.py`` loads the options file into a dictionary and calls
    ``CPET(options).run()``. Defaults are filled in and the options are checked
    when the object is created, and ``outputpath`` is created if needed.

    Parameters
    ----------
    options : dict
        Options for the calculation; must include ``CPET_method``. See the Options
        Reference.

    Attributes
    ----------
    options : dict
        The options, with defaults filled in.
    m : str
        The method to run, from ``CPET_method``.
    inputpath : str
        Folder of input files.
    outputpath : str
        Folder for results.
    inputfiletype : str
        ``"pdb"`` or ``"pqr"``.
    log : logging.Logger
        Logger for this module.
    """

    def __init__(self, options):
        # Logistics
        """
        Fill in defaults, check the options and create ``outputpath``.
        """
        self.options = default_options_initializer(options)
        check_valid_options(self.options)
        self.log = log
        self.m = self.options["CPET_method"]
        self.log.info(f"Instantiating CPET, method: {self.m}")
        self.inputpath = self.options["inputpath"]
        self.outputpath = self.options["outputpath"]
        self.inputfiletype = self.options["inputfiletype"]
        if not os.path.exists(self.outputpath):
            self.log.warning(
                f"Output directory does not exist, creating: {self.outputpath}"
            )
            os.makedirs(self.outputpath)

    def run(self):
        """
        Run the method named in ``CPET_method``.

        Raises
        ------
        ValueError
            If the method is not recognized.
        """
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
            raise ValueError(
                f"Method {self.m} not recognized. Please check the options file or the command line arguments."
            )

    def run_topo(self, gpu=False):
        """
        Compute the field topology of each structure in ``inputpath``.

        Structures are processed in random order and any with an existing
        ``<name>.top`` in ``outputpath`` are skipped, so several copies can run on
        the same folder at once. Each result is saved as ``<name>.top``.

        Parameters
        ----------
        gpu : bool, optional
            If True, use the GPU (``topo_GPU``); otherwise use CPU worker processes.
            Default is False.
        """
        if gpu:
            runtype = "compute_topo_GPU_batch_filter"
        else:
            runtype = "compute_topo_complete_c_shared"
        files_input = get_input_files(self.inputpath, filetype=self.inputfiletype)
        for i in range(len(files_input) * 100):
            if len(files_input) != 0:
                file = choice(files_input)
            else:
                self.log.info("No more files to process")
                break
            files_input.remove(file)
            protein = file.split("/")[-1].split(".")[0]
            self.log.info(f"Protein file: {protein}")
            files_done = [x for x in os.listdir(self.outputpath) if x.endswith("top")]
            if protein + ".top" not in files_done:
                self.calculator = calculator(self.options, path_to_pdb=file)
                hist = getattr(self.calculator, runtype)()
                np.savetxt(self.outputpath + f"/{protein}.top", hist)
            else:
                self.log.info(
                    f"Output already exists for protein {protein}, skipping"
                )

    def run_volume(self):
        """
        Compute the electric field on the grid for each structure in ``inputpath``.

        Structures are processed in random order and any with an existing
        ``<name>_efield.dat`` in ``outputpath`` are skipped, so several copies can run
        on the same folder at once. Each result is saved as ``<name>_efield.dat``.
        """

        files_input = get_input_files(self.inputpath, filetype=self.inputfiletype)

        for i in range(len(files_input) * 100):
            if len(files_input) != 0:
                file = choice(files_input)
            else:
                self.log.info("No more files to process")
                break
            protein = file.split("/")[-1].split(".")[0]
            files_input.remove(file)
            self.log.info(f"Protein file: {protein}")
            files_done = [
                x for x in os.listdir(self.outputpath) if x.endswith("_efield.dat")
            ]
            self.log.debug(f"Files done: {files_done}")
            if protein + "_efield.dat" not in files_done:
                self.calculator = calculator(self.options, path_to_pdb=file)
                field_box, mesh_shape = self.calculator.compute_box()
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
                    name=self.outputpath + f"/{protein}_efield.dat",
                    volume=field_box,
                    meta_data=meta_data,
                )
            else:
                self.log.info(
                    f"Output already exists for protein {protein}, skipping"
                )

    def run_point(self):
        """
        Compute the electric field, or its magnitude, at ``center`` for each structure.

        Writes one ``<name>:<value>`` line per structure to ``point_field.dat`` or
        ``point_mag.dat`` in ``outputpath``.
        """
        files_input = get_input_files(self.inputpath, filetype=self.inputfiletype)
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
                self.log.info(f"Protein file: {protein}")
                point_field_or_mag = getattr(self.calculator, runtype)()
                # Save the point field or magnitude to the output file
                f.write(f"{protein}:{point_field_or_mag}\n")

    def run_volume_ESP(self):
        """
        Compute the electrostatic potential on the grid for each structure in ``inputpath``.

        Structures are processed in random order and any with an existing
        ``<name>_esp.dat`` in ``outputpath`` are skipped, so several copies can run on
        the same folder at once. Each result is saved as ``<name>_esp.dat``.
        """
        files_input = get_input_files(self.inputpath, filetype=self.inputfiletype)
        for i in range(len(files_input) * 100):
            if len(files_input) != 0:
                file = choice(files_input)
            else:
                self.log.info("No more files to process")
                break
            protein = file.split("/")[-1].split(".")[0]
            files_input.remove(file)
            self.log.info(f"Protein file: {protein}")
            files_done = [
                x for x in os.listdir(self.outputpath) if x.endswith("_esp.dat")
            ]
            self.log.debug(f"Files done: {files_done}")
            if protein + "_esp.dat" not in files_done:
                self.calculator = calculator(self.options, path_to_pdb=file)
                esp_box, mesh_shape = self.calculator.compute_box_ESP()
                self.log.debug(f"ESP box shape: {esp_box.shape}")
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
                    name=self.outputpath + f"/{protein}_esp.dat",
                    volume=esp_box,
                    meta_data=meta_data,
                )
            else:
                self.log.info(
                    f"Output already exists for protein {protein}, skipping"
                )

    def run_box_check(self):
        """
        Log the atoms inside the box for each structure, to find clashes with the volume.

        Raises
        ------
        ValueError
            If ``filter_radius`` or ``filter_resnum`` is set, since they can't be used
            with this method.
        """
        files_input = get_input_files(self.inputpath, filetype=self.inputfiletype)
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
            self.log.info(f"Protein file: {protein}")
            report_inside_box(self.calculator)
        self.log.info("No more files to process")

    def run_cluster(self):
        """
        Cluster the topologies, fields or potentials in ``inputpath``. See :class:`CPET.source.cluster.cluster`.
        """
        self.log.info(f"Running cluster analysis, method: {self.m}")
        self.cluster = cluster(self.options)
        self.cluster.Cluster()

    def run_visualize_efield(self):
        """
        Write a BILD file for each field (``*_efield.dat``) or potential (``*_esp.dat``) file in ``inputpath``.
        """
        self.log.info(
            "Visualizing the electric field or electrostatic potential for use in Chimera/ChimeraX"
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
                    path_to_esp=file, outputpath=self.outputpath, options=self.options,
                )

    def run_pca(self):
        """
        Run PCA on the fields in ``inputpath`` (``pca``), or on each folder in ``inputpath_list`` (``pca_compare``).

        Raises
        ------
        ValueError
            If ``pca_compare`` is used without ``inputpath_list``.
        """
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
                self.log.warning(
                    "No outputpath_list provided, using defaults based on inputpath_list"
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
                    self.log.info(
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
