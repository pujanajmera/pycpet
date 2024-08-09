from CPET.source.calculator import calculator
from CPET.source.cluster import cluster
import CPET.utils.visualize as visualize
from CPET.source.pca import pca_pycpet
from CPET.utils.io import save_numpy_as_dat
from CPET.utils.calculator import report_inside_box
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
        
        if "step_size" in self.options:
            self.step_size = self.options["step_size"]
        if "dimensions" in self.options:
            self.dimesions = self.options["dimensions"]
        
        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)
        
        if self.m == "cluster" or self.m == "cluster_volume":
            # creates a cluster object
            self.cluster = cluster(options)
        
        if self.m == "pca": 
            # creates a pca object
            self.pca_pycpet = pca_pycpet(options)

        self.benchmark_samples = self.options["benchmark"]["n_samples"]
        self.benchmark_step_sizes = self.options["benchmark"]["step_size"]
        self.benchmark_replicas = self.options["benchmark"]["replicas"]
        self.profile = self.options["profile"]


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
        elif self.m == "box_check":
            self.run_box_check()
        elif self.m == "visualize_field":
            self.run_visualize_efield()
        elif self.m == "pca": 
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
                    "dimensions": self.dimesions, 
                    "step_size": [self.step_size, self.step_size, self.step_size],
                    "num_steps": [mesh_shape[0], mesh_shape[1], mesh_shape[2]],
                    "transformation_matrix": self.calculator.transformation_matrix,
                    "center": self.calculator.center,
                }
                
                save_numpy_as_dat(
                    name=self.outputpath + "/{}_efield.dat".format(protein),
                    field=field_box,
                    meta_data=meta_data
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
                x for x in os.listdir(self.outputpath) if x[-11:] == "_efield.dat"
            ]
            if protein + ".top" not in files_done:
                field_box = self.calculator.compute_box_ESP()
                np.savetxt(
                    self.outputpath + "/{}_esp.dat".format(protein),
                    field_box,
                    fmt="%.3f",
                )

    def run_box_check(self, num=100000):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        for file in files_input:
            if "filter_radius" in self.options or "filter_resids" in self.options or "filter_resnum" in self.options:
                #Error out, radius not compatible
                raise ValueError(
                    "filter_radius/filter_resids/filter_resnum is not compatible with box_check. Please remove from options"
                    )
            #Need to not filter in box to check, but can filter all else
            self.options["filter_in_box"] = False
            self.calculator = calculator(self.options, path_to_pdb=file)
            protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
            print("protein file: {}".format(protein))
            report_inside_box(self.calculator)
        print("No more files to process!")

    def run_cluster(self):
        self.cluster.Cluster()

    def run_visualize_efield(self):
        visualize.visualize_fields(self.inputpath, self.outputpath, self.options)

    def run_pca(self):
        _, _ = self.pca_pycpet.fit_and_transform()
