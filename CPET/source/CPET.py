from CPET.source.calculator import calculator
from CPET.source.cluster import cluster
from CPET.source.benchmark import gen_param_dist_mat, analyze_param_dist_mat
from CPET.utils.calculator import make_histograms, construct_distance_matrix
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
        self.cluster = cluster(options)
        self.benchmark_samples = self.options["benchmark"]["n_samples"]
        self.benchmark_step_sizes = self.options["benchmark"]["step_size"]
        self.benchmark_replicas = self.options["benchmark"]["replicas"]
    
    def run(self):
        if self.m == "topo":
            self.run_topo()
        elif self.m == "topo_GPU":
            self.run_topo_GPU()
        elif self.m == "volume":
            self.run_volume()
        elif self.m == "point_field":
            self.run_point_field()
        elif self.m == "point_mag":
            self.run_point_mag()
        elif self.m == "benchmark":
            self.run_benchmarking()
        elif self.m == "cluster":
            self.run_cluster()
    
    def run_topo(self, num=50000, benchmarking=False):
        files_input = glob(self.inputpath + "/*.pdb")
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
                hist = self.calculator.compute_topo()
                if not benchmarking:
                    np.savetxt("{}.top".format(protein), hist)
                if benchmarking:
                    np.savetxt("{}_{}_{}_{}.top".format(protein, self.calculator.n_samples, str(self.calculator.step_size)[2:], self.replica), hist)
    
    def run_topo_GPU(self, num=50000, benchmarking=False):
        files_input = glob(self.inputpath + "/*.pdb")
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
    
    def run_volume(self, num=50000):
        files_input = glob(self.inputpath + "/*.pdb")
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
            files_done = [x for x in os.listdir(self.outputpath) if x[-11:]=="_efield.dat"]
            if protein+".top" not in files_done:
                field_box = self.calculator.compute_box()
                np.savetxt(self.outputpath + "{}_efield.dat".format(protein), field_box)
  
    def run_point_field(self):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        outfile = self.outputpath + "/point_field.dat"
        with open(outfile, "w") as f:        
            for file in files_input:
                self.calculator = calculator(self.options, path_to_pdb = file)
                protein = file.split("/")[-1].split(".")[0]
                print("protein file: {}".format(protein))
                point_field = self.calculator.compute_point_field()
                f.write("{}:{}\n".format(protein,point_field))

    def run_point_mag(self):
        files_input = glob(self.inputpath + "/*.pdb")
        if len(files_input) == 0:
            raise ValueError("No pdb files found in the input directory")
        if len(files_input) == 1:
            warnings.warn("Only one pdb file found in the input directory")
        outfile = self.outputpath + "/point_mag.dat"
        with open(outfile, "w") as f:
            for file in files_input:
                self.calculator = calculator(self.options, path_to_pdb = file)
                protein = file.split("/")[-1].split(".")[0]
                print("protein file: {}".format(protein))
                point_field = self.calculator.compute_point_mag()
                f.write("{}:{}\n".format(protein,point_field))

    def run_benchmarking(self):
        files_input = glob(self.inputpath + "/*.pdb")
        num=5
        if len(files_input) < 5:
            warnings.warn("Less than 5 pdb files found in the input directory, benchmarking on {} files. This may be insufficient sampling".format(len(files_input)))
            num = len(files_input)
        if len(files_input) > 5:
            warnings.warn("More than 5 pdb files found in the input directory, choosing 5 random pdbs to benchmarking on")
            files_input = [choice(files_input) for i in range(num)]
        
        for step_size in self.benchmark_step_sizes:
            for n_samples in self.benchmark_samples:
                for i in range(self.benchmark_replicas):
                    self.replica = i
                    for file in files_input:
                        self.calculator = calculator(self.options, path_to_pdb = file)
                        self.calculator.n_samples = n_samples
                        self.calculator.step_size = step_size
                        protein = self.calculator.path_to_pdb.split("/")[-1].split(".")[0]
                        hist = self.calculator.compute_topo_batched()
                        np.savetxt("{}_{}_{}_{}.top".format(protein, self.calculator.n_samples, str(self.calculator.step_size)[2:], self.replica), hist)
        for file in files_input:
            topo_files = glob(self.outputpath + "/{}*.top".format(file.split("/")[-1].split(".")[0]))
            if len(topo_files) != num*len(self.benchmark_samples)*len(self.benchmark_step_sizes):
                raise ValueError("Incorrect number of output topologies for requested benchmark parameters")
            histograms = make_histograms(topo_files)
            distance_matrix = construct_distance_matrix(histograms)
            avg_dist = gen_param_dist_mat(distance_matrix, topo_files)
            analyze_param_dist_mat(avg_dist, threshold=self.options["benchmark"]["threshold"], mode=self.options["benchmark"]["mode"])        
        
    def run_cluster(self):
        self.cluster.Cluster()