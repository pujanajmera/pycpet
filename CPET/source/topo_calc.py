import numpy as np
import time
from multiprocessing import Pool


from CPET.utils.parser import parse_pqr
from CPET.utils.c_ops import Math_ops
from CPET.utils.parallel import task, task_batch
from CPET.utils.calculator import initialize_box_points
from CPET.utils.fastmath import nb_subtract, power, nb_norm, nb_cross
from CPET.source.calculator import calculator
from CPET.utils.parser import filter_pqr_radius, filter_pqr_residue

class Topo_calc:
    def __init__(self, options):
        #self.efield_calc = calculator(math_loc=math_loc)
        self.options = options
        
        self.path_to_pqr = options["path_to_pqr"]
        self.center = np.array(options["center"])
        self.x_vec_pt = np.array(options["x"])
        self.y_vec_pt = np.array(options["y"])
        self.dimensions = np.array(options["dimensions"])
        self.step_size = options["step_size"]
        self.n_samples = options["n_samples"]
        self.concur_slip = options["concur_slip"]
        if "batch_size" in options.keys(): 
            self.batch_size = options["batch_size"]
            
        
        if "filter_resids" in options.keys(): 
            print("filtering residues: {}".format(options["filter_resids"]))
            self.x, self.Q, self.resids = parse_pqr(self.path_to_pqr, ret_residue_names=True)
            self.x, self.Q = filter_pqr_residue(self.x, self.Q, self.resids, filter_list=options["filter_resids"])
            
        else: 
            self.x, self.Q = parse_pqr(self.path_to_pqr)


        if "filter_radius" in options.keys():
            print("filtering by radius: {} Ang".format(options["filter_radius"]))
            
            self.x, self.Q = filter_pqr_radius(
                x=self.x, 
                Q=self.Q, 
                center=self.center, 
                radius=options["filter_radius"])
            
        (
            self.random_start_points,
            self.random_max_samples,
            self.transformation_matrix,
        ) = initialize_box_points(
            self.center,
            self.x_vec_pt,
            self.y_vec_pt,
            self.dimensions,
            self.n_samples,
            self.step_size,
        )



        self.x = (self.x - self.center) @ np.linalg.inv(self.transformation_matrix)
        
        if "batch_size" in options.keys(): 
            self.batch_size = options["batch_size"]
            #self.x_batched = 
        
        print("... > Initialized Topo_calc!")

    def compute_topo(self):
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            hist = pool.starmap(task, args)
        end_time = time.time()
        self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist


    def compute_topo_batched(self):
        print("... > Computing Topo in Batches!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            hist = pool.starmap(task, args)
        end_time = time.time()
        self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist


