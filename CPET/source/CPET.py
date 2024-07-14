from CPET.source.calculator import calculator
from CPET.source.cluster import cluster
import CPET.utils.visualize as visualize
from glob import glob
from random import choice
import os
import numpy as np
import warnings


class CPET:
    def __init__(self, options):
        self.options = options
        self.m = self.options["CPET_method"]
        self.inputpath = self.options["inputpath"]
        self.outputpath = self.options["outputpath"]
        self.benchmark_samples = self.options["benchmark"]["n_samples"]
        self.benchmark_step_sizes = self.options["benchmark"]["step_size"]
        self.benchmark_replicas = self.options["benchmark"]["replicas"]
        self.profile = self.options["profile"]
        if self.m == "cluster" or self.m == "cluster_volume":
            self.cluster = cluster(options)
        if self.m == "visualize_field":
            self.visualize = visualize(options)

    def run(self):
        if self.m == "topo":
            self.run_topo()
        elif self.m == "topo_GPU":
            self.run_topo_GPU()
        elif self.m == "topo_griddev":
            self.run_topo_griddev()
        elif self.m == "mtopo":
            self.run_mtopo()  # For dev primarily
        elif self.m == "volume":
            self.run_volume()
        elif self.m == "volume_ESP":
            self.run_volume_ESP()
        elif self.m == "point_field":
            self.run_point_field()
        elif self.m == "point_mag":
            self.run_point_mag()
        elif self.m == "cluster" or self.m == "cluster_volume":
            self.run_cluster()
        elif self.m == "visualize_field":
            self.run_visualize_field()
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
            file = choice(files_input)
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            files_input.remove(file)
            print("protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x.split(".")[-1] == "top"
            ]
            if protein + ".top" not in files_done:
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
                hist = (
                    self.calculator.compute_topo_GPU_batch_filter()
                )
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

    def run_topo_griddev(self, num=100000, benchmarking=False):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for i in range(num):
            file = choice(files_input)
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            files_input.remove(file)
            print("protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x.split(".")[-1] == "top"
            ]
            if protein + ".top" not in files_done:
                hist = self.calculator.compute_topo_griddev()
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

    def run_topo_mtopo(self, num=100000, benchmarking=False):
        """files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for i in range(num):
            file = choice(files_input)
            self.calculator = calculator(self.options, path_to_pdb = file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            files_input.remove(file)
            print("protein file: {}".format(protein))
            files_done = [x for x in os.listdir(self.outputpath) if x.split(".")[-1]=="top"]
            if protein+".top" not in files_done:
                hist = self.calculator.compute_topo_batch_filter()
                if not benchmarking:
                    np.savetxt(self.outputpath + "{}.top".format(protein), hist)
                if benchmarking:
                    np.savetxt(self.outputpath + "{}_{}_{}_{}.top".format(protein, self.calculator.n_samples, str(self.calculator.step_size)[2:], self.replica), hist)
        """
        return "You have reached the limit of this package's capabilities at the moment, we do not support the function called as of yet"

    def run_volume(self, num=100000):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for i in range(num):
            file = choice(files_input)
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            files_input.remove(file)
            print("protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x[-11:] == "_efield.dat"
            ]
            if protein + ".top" not in files_done:
                field_box = self.calculator.compute_box()
                np.savetxt(
                    self.outputpath + "/{}_efield.dat".format(protein),
                    field_box,
                    fmt="%.3f",
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
            file = choice(files_input)
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            files_input.remove(file)
            print("protein file: {}".format(protein))
            files_done = [
                x for x in os.listdir(self.outputpath) if x[-11:] == "_efield.dat"
            ]
            if protein + ".top" not in files_done:
                field_box = self.calculator.compute_box_ESP()
                np.savetxt(
                    self.outputpath + "/{}_esp.dat".format(protein),
                    field_box,
                    fmt="%.3f",
                )

    def run_cluster(self):
        self.cluster.Cluster()

    def run_visualize_efield(self):
        visualize.visualize_fields(self.inputpath, self.outputpath, self.options)
