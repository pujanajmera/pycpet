""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Dev script to benchmark the performance of various topo calc methods
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import time
from multiprocessing import Pool


from CPET.utils.parser import parse_pdb
from CPET.utils.c_ops import Math_ops
from CPET.utils.parallel import task, task_batch, task_base
from CPET.utils.calculator import initialize_box_points_random, initialize_box_points_uniform, compute_field_on_grid, calculate_electric_field_dev_c_shared
from CPET.utils.parser import parse_pdb, filter_radius, filter_residue, filter_in_box, calculate_center, filter_resnum
from CPET.utils.gpu import compute_curv_and_dist_mat_gpu, propagate_topo_matrix_gpu, batched_filter_gpu, initialize_streamline_grid_gpu

class calculator:
    def __init__(self, options, path_to_pdb = None):
        #self.efield_calc = calculator(math_loc=math_loc)
        self.options = options
        self.dimensions = np.array(options["dimensions"])
        self.step_size = options["step_size"]
        self.n_samples = options["n_samples"]
        self.concur_slip = options["concur_slip"]
        self.path_to_pdb = path_to_pdb

        if "GPU_batch_freq" in options.keys():
            self.GPU_batch_freq = options["GPU_batch_freq"]
        else:
            self.GPU_batch_freq = 100
            
        self.x, self.Q, self.atom_number, self.resids, self.residue_number, self.atom_type = parse_pdb(
            self.path_to_pdb, get_charges=True)        

        ##################### define center axis
        
        if type(options["center"]) == list:
            self.center = np.array(options["center"])

        elif type(options["center"]) == dict:
            method = options["center"]["method"]
            centering_atoms = [(element, options["center"]["atoms"][element]) for element in options["center"]["atoms"]]
            pos_considered = [
                pos for atom_res in centering_atoms
                for idx, pos in enumerate(self.x)
                if (self.atom_type[idx], self.residue_number[idx]) == atom_res
            ]
            self.center = calculate_center(pos_considered, method=method)
        else: 
            raise ValueError("center must be a list or dict")
        
        ##################### define x axis

        if type(options["x"]) == list:
            self.x_vec_pt = np.array(options["x"])
        
        elif type(options["x"]) == dict:
            method = options["x"]["method"]
            centering_atoms = [(element, options["x"]["atoms"][element]) for element in options["x"]["atoms"]]
            pos_considered = [
                pos for atom_res in centering_atoms
                for idx, pos in enumerate(self.x)
                if (self.atom_type[idx], self.residue_number[idx]) == atom_res
            ]
            self.x_vec_pt = calculate_center(pos_considered, method=method)
        
        else: 
            raise ValueError("x must be a list or dict")
        
        ##################### define y axis
        
        if type(options["y"]) == list:
            self.y_vec_pt = np.array(options["y"])
        
        elif type(options["y"]) == dict:
            method = options["y"]["method"]
            centering_atoms = [(element, options["y"]["atoms"][element]) for element in options["y"]["atoms"]]
            pos_considered = [
                pos for atom_res in centering_atoms
                for idx, pos in enumerate(self.x)
                if (self.atom_type[idx], self.residue_number[idx]) == atom_res
            ]
            self.y_vec_pt = calculate_center(pos_considered, method=method)

        else:
            raise ValueError("y must be a list or dict")

        print(len(self.Q))
        if "filter_resids" in options.keys(): 
            #print("filtering residues: {}".format(options["filter_resids"]))                
            self.x, self.Q = filter_residue(
                self.x, self.Q, self.resids, filter_list=options["filter_resids"])
            
        if "filter_resnum" in options.keys(): 
            #print("filtering residues: {}".format(options["filter_resids"]))                
            self.x, self.Q = filter_resnum(
                self.x, self.Q, self.residue_number, filter_list=options["filter_resnum"])

        if "filter_radius" in options.keys():
            print("filtering by radius: {} Ang".format(options["filter_radius"]))
            
            r = np.linalg.norm(self.x, axis=1)
            #print("r {}".format(r))
            
            self.x, self.Q = filter_radius(
                x=self.x, 
                Q=self.Q, 
                center=self.center, 
                radius=float(options["filter_radius"]))
            
            #print("center {}".format(self.center))
            r = np.linalg.norm(self.x, axis=1)
            #print("r {}".format(r))

        if "filter_in_box" in options.keys():
            if bool(options["filter_in_box"]):
                #print("filtering charges in sampling box")
                self.x, self.Q = filter_in_box(x=self.x, Q=self.Q, center=self.center,  dimensions=self.dimensions)

        (
            self.random_start_points,
            self.random_max_samples,
            self.transformation_matrix,
        ) = initialize_box_points_random(
            self.center,
            self.x_vec_pt,
            self.y_vec_pt,
            self.dimensions,
            self.n_samples,
            self.step_size,
        )

        '''(
            self.mesh, self.uniform_transformation_matrix
        ) = initialize_box_points_uniform(
            center=self.center,
            x=self.x_vec_pt,
            y=self.y_vec_pt,
            dimensions=self.dimensions,
            step_size=self.step_size,
        )'''

        #self.transformation_matrix and self.uniform_transformation_matrix are the same

        self.x = (self.x - self.center) @ np.linalg.inv(self.transformation_matrix)
        #print(self.random_start_points)
        
        if "batch_size" in options.keys(): 
            self.batch_size = options["batch_size"]
            # reshape the random_start_points and random_max_samples to be batches of size batch_size
            self.random_start_points_batched = np.array(self.random_start_points).reshape(int(self.n_samples/self.batch_size), self.batch_size, 3)
            self.random_max_samples_batched = np.array(self.random_max_samples).reshape(int(self.n_samples/self.batch_size), self.batch_size)
            
            #self.random_max_samples_batched = [self.batch_size for i in range(self.n_samples//self.batch_size)]
            #self.random_max_samples_batched.append(self.n_samples%self.batch_size)
            #self.random_start_points_batched = [i*self.batch_size for i in range(self.n_samples//self.batch_size)]
            #self.random_start_points_batched.append((self.n_samples//self.batch_size)*self.batch_size)

        print("... > Initialized Calculator!")


    def compute_topo_base(self): 
        
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        #print("starting pooling")
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            raw = pool.starmap(task_base, args)
            #print(raw)
            dist = [i[0] for i in raw]
            curve = [i[1] for i in raw]
            hist=[dist, curve]
        end_time = time.time()
        self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist
    
    
    def compute_topo(self):
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        #print("starting pooling")
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            #raw = pool.starmap(task, args)
            
            result = pool.starmap_async(task, args)
            raw = []
            for result in result.get():
                raw.append(result)
            
            dist = [i[0] for i in raw]
            curve = [i[1] for i in raw]
            hist=[dist, curve]
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
        print("num batches: {}".format(len(self.random_start_points_batched)))
        #print(self.random_start_points_batched)
        #print(self.random_max_samples_batched)
        with Pool(self.concur_slip) as pool:
            args = [
                    (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                    for i, n_iter in zip(self.random_start_points_batched, self.random_max_samples_batched)

            ]
            raw = pool.starmap(task_batch, args)

            '''result = pool.starmap_async(task_batch, args)
            raw = []
            for result in result.get():
                raw.append(result)'''
            # reshape 
            # print(raw)
            hist = np.array(raw).reshape(self.n_samples, 2)
            dist = hist[0, :]
            curv = hist[1, :]
            hist = [dist, curv]
            
        end_time = time.time()
        #self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist


    def compute_box(self):
        print("... > Computing Box!")
        print(f"Number of charges: {len(self.Q)}")
        print("mesh shape: {}".format(self.mesh.shape))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        field_box = compute_field_on_grid(self.mesh, self.x, self.Q)
        return field_box

    def compute_point_mag(self):
        print("... > Computing Point Magnitude!")
        print(f"Number of charges: {len(self.Q)}")
        print("point: {}".format(self.center))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        start_time = time.time()
        point_mag = np.norm(calculate_electric_field_dev_c_shared(self.x, self.Q, self.center))
        end_time = time.time()
        print(f"{end_time - start_time:.2f}")
        return point_mag
    
    def compute_point_field(self):
        print("... > Computing Point Field!")
        print(f"Number of charges: {len(self.Q)}")
        print("point: {}".format(self.center))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        start_time = time.time()
        point_field = calculate_electric_field_dev_c_shared(self.x, self.Q, self.center)
        end_time = time.time()
        print(f"{end_time - start_time}")
        return point_field


    def compute_topo_batch_filter(self):
        import torch

        print("... > Computing Topo in Batches!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        

        self.x_vec_pt
        self.x
        self.y_vec_pt
        self.dimensions
        self.step_size
        self.n_samples
        self.GPU_batch_freq

        Q_gpu = torch.tensor(self.Q).cuda()
        x_gpu = torch.tensor(self.x).cuda()
        dim_gpu = torch.tensor(self.dimensions).cuda()
        step_size_gpu = torch.tensor([self.step_size]).cuda()

        path_matrix, _, M, path_filter, _ = initialize_streamline_grid_gpu(self.center, self.x_vec_pt, self.y_vec_pt, self.dimensions, self.n_samples, self.step_size)
        path_matrix_torch=torch.tensor(path_matrix).cuda()
        path_filter=torch.tensor(path_filter).cuda()
        dumped_values=torch.tensor(np.empty((6,0,3))).cuda()
        start_time = time.time()
        j=0
        start_time = time.time()
        for i in range(len(path_matrix)):
            '''if i % 100 == 0:
                print(i)'''

            if(j == len(path_matrix)-1):
                break
            #path_matrix_torch = propagate_topo_matrix_gpu(path_matrix_torch, torch.tensor([i]).cuda(), x_gpu, Q_gpu, step_size_gpu)
            path_matrix_torch = propagate_topo_matrix_gpu(path_matrix_torch, i, x_gpu, Q_gpu, self.step_size)
            if(i%self.GPU_batch_freq == 0 and i>5):
                path_matrix_torch, dumped_values, path_filter= batched_filter_gpu(path_matrix_torch, dumped_values, i, dim_gpu, M, path_filter, current=True)
                #GPU_batch_freq *= 2
            j += 1
            torch.cuda.empty_cache()
            if dumped_values.shape[1]>=self.n_samples:
                break
        print(path_matrix_torch.shape)
        print(dumped_values.shape)
        distances, curvatures = compute_curv_and_dist_mat_gpu(dumped_values[0,:,:], dumped_values[1,:,:], dumped_values[2,:,:],dumped_values[3,:,:],dumped_values[4,:,:],dumped_values[5,:,:])
    
        end_time = time.time()
        print(f"Time taken for {self.n_samples} calculations with N~{self.Q.shape}: {end_time - start_time:.2f} seconds")
        topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
        return topology
