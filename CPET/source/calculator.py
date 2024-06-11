""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Dev script to benchmark the performance of various topo calc methods
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import time
from multiprocessing import Pool
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.jit as jit


from CPET.utils.parser import parse_pdb
from CPET.utils.c_ops import Math_ops
from CPET.utils.parallel import task, task_batch, task_base
from CPET.utils.calculator import initialize_box_points_random, initialize_box_points_uniform, compute_field_on_grid, calculate_electric_field_dev_c_shared, compute_ESP_on_grid
from CPET.utils.parser import parse_pdb, filter_radius, filter_residue, filter_in_box, calculate_center, filter_resnum, filter_resnum_andname
from CPET.utils.gpu import compute_curv_and_dist_mat_gpu, propagate_topo_matrix_gpu, batched_filter_gpu, initialize_streamline_grid_gpu, batched_filter_gpu_end
from CPET.utils.gridcpu import compute_curv_and_dist_mat_gridcpu, propagate_topo_matrix_gridcpu, batched_filter_gridcpu, initialize_streamline_grid_gridcpu
from CPET.utils.gpu_alt import compute_curv_and_dist_mat_gpu_alt, propagate_topo_matrix_gpu_alt, batched_filter_gpu_alt, batched_filter_gpu_end_alt, initialize_streamline_grid_gpu_alt

class calculator:
    def __init__(self, options, path_to_pdb = None):
        #self.efield_calc = calculator(math_loc=math_loc)
        self.options = options
        self.dimensions = np.array(options["dimensions"])
        self.step_size = options["step_size"]
        self.n_samples = options["n_samples"]
        self.concur_slip = options["concur_slip"]
        self.profile = options["profile"]
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
            
        if "filter_resnum_andname" in options.keys():
            #print("filtering residues: {}".format(options["filter_resids"]))                
            self.x, self.Q = filter_resnum_andname(
                self.x, self.Q, self.residue_number, self.resids, filter_list=options["filter_resnum_andname"])

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

        if options["CPET_method"] == "topo_GPU" or options["CPET_method"] == "topo" or options["CPET_method"] == "topo_griddev" or options["CPET_method"] == "benchmark":
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
        
        elif options["CPET_method"] == "volume" or options["CPET_method"] == "volume_ESP":
            (
                self.mesh, self.transformation_matrix
            ) = initialize_box_points_uniform(
                center=self.center,
                x=self.x_vec_pt,
                y=self.y_vec_pt,
                dimensions=self.dimensions,
                step_size=self.step_size,
            )

        #self.transformation_matrix and self.uniform_transformation_matrix are the same

        self.x = (self.x - self.center) @ np.linalg.inv(self.transformation_matrix)
        #print(self.random_start_points)
        
        if "batch_size" in options.keys() and options["CPET_method"] == "woohoo": 
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

    def compute_box_ESP(self):
        print("... > Computing Box!")
        print(f"Number of charges: {len(self.Q)}")
        print("mesh shape: {}".format(self.mesh.shape))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        print("Transformation matrix: {}".format(self.transformation_matrix))
        print("Center: {}".format(self.center))
        field_box = compute_ESP_on_grid(self.mesh, self.x, self.Q)
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


    def compute_topo_GPU_batch_filter(self):
        if self.profile == True:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True) as prof:
                num_per_dim = round(self.n_samples ** (1/3))
                if num_per_dim ** 3 < self.n_samples:
                    num_per_dim += 1
                self.n_samples = num_per_dim ** 3
                print("... > Computing Topo in Batches on a GPU!")
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

                '''
                Q_gpu = torch.tensor(self.Q, dtype=torch.float16).cuda()
                Q_gpu = Q_gpu.unsqueeze(0)
                x_gpu = torch.tensor(self.x, dtype=torch.float16).cuda()
                dim_gpu = torch.tensor(self.dimensions, dtype=torch.float16).cuda()
                step_size_gpu = torch.tensor([self.step_size], dtype=torch.float16).cuda()
                '''

                Q_gpu = torch.tensor(self.Q, dtype=torch.float32).cuda()
                Q_gpu = Q_gpu.unsqueeze(0)
                x_gpu = torch.tensor(self.x, dtype=torch.float32).cuda()
                dim_gpu = torch.tensor(self.dimensions, dtype=torch.float32).cuda()
                step_size_gpu = torch.tensor([self.step_size], dtype=torch.float32).cuda()

                path_matrix, _, M, path_filter, _ = initialize_streamline_grid_gpu(self.center, self.x_vec_pt, self.y_vec_pt, self.dimensions, num_per_dim, self.step_size)
                
                '''
                path_matrix_torch=torch.tensor(path_matrix, dtype=torch.float16).cuda()
                path_filter=torch.tensor(path_filter, dtype=torch.float16).cuda()
                dumped_values=torch.tensor(np.empty((6,0,3)), dtype=torch.float16).cuda()
                '''
                path_matrix_torch=torch.tensor(path_matrix, dtype=torch.float32).cuda()
                path_filter=torch.tensor(path_filter, dtype=torch.float32).cuda()
                dumped_values=torch.tensor(np.empty((6,0,3)), dtype=torch.float32).cuda()

                start_time = time.time()
                j=0
                start_time = time.time()
                for i in range(len(path_matrix)):
                    if i % 100 == 0:
                        print(i)

                    if(j == len(path_matrix)-1):
                        break
                    path_matrix_torch = propagate_topo_matrix_gpu(path_matrix_torch, torch.tensor([i]).cuda(), x_gpu, Q_gpu, step_size_gpu)
                    #path_matrix_torch = propagate_topo_matrix_gpu(path_matrix_torch, i, x_gpu, Q_gpu, self.step_size)
                    if(i%self.GPU_batch_freq == 0 and i>5):
                        path_matrix_torch, dumped_values, path_filter= batched_filter_gpu(path_matrix_torch, dumped_values, i, dim_gpu, M, path_filter, current=True)
                        #GPU_batch_freq *= 2
                    j += 1
                    if dumped_values.shape[1]>=self.n_samples:
                        break
                torch.cuda.empty_cache()
                path_matrix_torch, dumped_values, path_filter= batched_filter_gpu_end(path_matrix_torch, dumped_values, i, dim_gpu, M, path_filter, current=True)
                print(path_matrix_torch.shape)
                print(dumped_values.shape)
                distances, curvatures = compute_curv_and_dist_mat_gpu(dumped_values[0,:,:], dumped_values[1,:,:], dumped_values[2,:,:],dumped_values[3,:,:],dumped_values[4,:,:],dumped_values[5,:,:])
            
                end_time = time.time()
                print(f"Time taken for {self.n_samples} calculations with N~{self.Q.shape}: {end_time - start_time:.2f} seconds")
                topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            prof.export_chrome_trace("trace.json")  # Optional: Export trace to view it in Chrome's trace viewer
                

        else:
            num_per_dim = round(self.n_samples ** (1/3))
            if num_per_dim ** 3 < self.n_samples:
                num_per_dim += 1
            self.n_samples = num_per_dim ** 3
            print("... > Computing Topo in Batches on a GPU!")
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

            Q_gpu = torch.tensor(self.Q, dtype=torch.float32).cuda()
            Q_gpu = Q_gpu.unsqueeze(0)
            x_gpu = torch.tensor(self.x, dtype=torch.float32).cuda()
            dim_gpu = torch.tensor(self.dimensions, dtype=torch.float32).cuda()
            step_size_gpu = torch.tensor([self.step_size], dtype=torch.float32).cuda()
            
            
            path_matrix, _, M, path_filter, _ = initialize_streamline_grid_gpu(self.center, self.x_vec_pt, self.y_vec_pt, self.dimensions, num_per_dim, self.step_size)

            path_matrix_torch=torch.tensor(path_matrix, dtype=torch.float32).cuda()
            path_filter=torch.tensor(path_filter, dtype=torch.bool).cuda()
            dumped_values=torch.tensor(np.empty((6,0,3)), dtype=torch.float32).cuda()
            

            start_time = time.time()
            j=0
            start_time = time.time()
            for i in range(len(path_matrix)):
                if i % 100 == 0:
                    print(i)
                if(j == len(path_matrix)-1):
                    break
                path_matrix_torch = propagate_topo_matrix_gpu(path_matrix_torch, torch.tensor([i]).cuda(), x_gpu, Q_gpu, step_size_gpu)
                
                #path_matrix_torch = propagate_topo_matrix_gpu(path_matrix_torch, i, x_gpu, Q_gpu, self.step_size)
                if(i%self.GPU_batch_freq == 0 and i>5):
                    
                    path_matrix_torch, dumped_values, path_filter= batched_filter_gpu(path_matrix_torch, dumped_values, i, dim_gpu, M, path_filter, current=True)
                    #GPU_batch_freq *= 2
                j += 1
                if dumped_values.shape[1]>=self.n_samples:
                    break
            torch.cuda.empty_cache()
            path_matrix_torch, dumped_values, path_filter= batched_filter_gpu(path_matrix_torch, dumped_values, i, dim_gpu, M, path_filter, current=True)
            print(path_matrix_torch.shape)
            print(dumped_values.shape)
            np.savetxt("dumped_values_init.txt", dumped_values[0:3].cpu().numpy().transpose(1, 0, 2).reshape(dumped_values.shape[1], -1), fmt='%.6f')
            np.savetxt("dumped_values_final.txt", dumped_values[3:6].cpu().numpy().transpose(1, 0, 2).reshape(dumped_values.shape[1], -1), fmt='%.6f')
            distances, curvatures = compute_curv_and_dist_mat_gpu(dumped_values[0,:,:], dumped_values[1,:,:], dumped_values[2,:,:],dumped_values[3,:,:],dumped_values[4,:,:],dumped_values[5,:,:])
            print(distances)
            print(curvatures)
            end_time = time.time()
            print(f"Time taken for {self.n_samples} calculations with N~{self.Q.shape}: {end_time - start_time:.2f} seconds")
            topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
            print(topology.shape)
        return topology
    

    def compute_topo_griddev(self):
        print("... > Computing Grid Topo in Batches! -- IN DEV")
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

        path_matrix, _, M, path_filter, _ = initialize_streamline_grid_gridcpu(self.center, self.x_vec_pt, self.y_vec_pt, self.dimensions, self.n_samples, self.step_size)
        dumped_values=np.empty((6,0,3))
        start_time = time.time()
        j=0
        start_time = time.time()
        for i in range(len(path_matrix)):
            if i % 100 == 0:
                print(i)

            if(j == len(path_matrix)-1):
                break
            #start_time_prop = time.time()
            path_matrix = propagate_topo_matrix_gridcpu(path_matrix, i, self.x, self.Q, self.step_size)
            #end_time_prop = time.time()
            #print(f"Time taken for propagation: {end_time_prop - start_time_prop:.2f} seconds")
            if(i%self.GPU_batch_freq == 0 and i>5):
                #start_time_filter = time.time()
                path_matrix, dumped_values, path_filter= batched_filter_gridcpu(path_matrix, dumped_values, i, self.dimensions, M, path_filter, current=True)
                #end_time_filter = time.time()
                #print(f"Time taken for filtering: {end_time_filter - start_time_filter:.2f} seconds")
                #GPU_batch_freq *= 2
            j += 1
            if dumped_values.shape[1]>=self.n_samples:
                break
        path_matrix, dumped_values, path_filter= batched_filter_gridcpu(path_matrix, dumped_values, i, self.dimensions, M, path_filter, current=True)
        print(path_matrix.shape)
        print(dumped_values.shape)

        #start_time_distcurv = time.time()
        distances, curvatures = compute_curv_and_dist_mat_gridcpu(dumped_values[0,:,:], dumped_values[1,:,:], dumped_values[2,:,:],dumped_values[3,:,:],dumped_values[4,:,:],dumped_values[5,:,:])
        #end_time_distcurv = time.time()
        #print(f"Time taken for distance and curvature calculation: {end_time_distcurv - start_time_distcurv:.2f} seconds")
    
        end_time = time.time()
        print(f"Time taken for {self.n_samples} calculations with N~{self.Q.shape}: {end_time - start_time:.2f} seconds")
        topology = np.column_stack((distances, curvatures))
        return topology


    def compute_topo_GPU_batch_filter_alt(self):
        if self.profile == True:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True) as prof:
                num_per_dim = round(self.n_samples ** (1/3))
                if num_per_dim ** 3 < self.n_samples:
                    num_per_dim += 1
                self.n_samples = num_per_dim ** 3
                print("... > Computing Topo in Batches on a GPU!")
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



                path_matrix, _, M, path_filter, _, x_gpu, Q_gpu, dim_gpu, step_size_gpu = initialize_streamline_grid_gpu_alt(self.center, self.x_vec_pt, self.y_vec_pt, self.dimensions, num_per_dim, self.step_size, self.GPU_batch_freq, self)

                path_matrix_torch=torch.tensor(path_matrix, dtype=torch.float32).cuda() #Of shape (GPU_batch_freq, n_samples, 3)
                path_filter=torch.tensor(path_filter, dtype=torch.float32).cuda() #Of shape (GPU_batch_freq, n_samples, 1)
                dumped_values=torch.tensor(np.empty((3,0,3)), dtype=torch.float32).cuda() #Of shape (3,0,3)

                max_num_batch = int((M+2-2)/(self.GPU_batch_freq-2)) #+2 since M+2 points need to be computed, and -2 since first two initial points are already computed, and propagations of 98 steps
                remainder = (M+2-2)%(self.GPU_batch_freq-2)
                #print(path_matrix_torch.shape)
                #print(dumped_values.shape)
                #print(path_filter.shape)
                #For streamline points not including remainder
                start_time = time.time()
                for i in range(max_num_batch):
                    print(i)
                    for j in range(self.GPU_batch_freq-2):
                        path_matrix_torch = propagate_topo_matrix_gpu_alt(path_matrix_torch, torch.tensor([j+1]).cuda(), x_gpu, Q_gpu, step_size_gpu)
                        if i==0 and j==0:
                            init_points = path_matrix_torch[0:3,...]
                    print("filtering!")
                    #print(path_matrix_torch[:,0,:])
                    #print(path_filter[:,0,:])
                    path_matrix_torch, dumped_values, path_filter= batched_filter_gpu_alt(path_matrix_torch, dumped_values, i, dim_gpu, M, path_filter, self.GPU_batch_freq, current=True)
                    #print(path_matrix_torch[:,0,:])
                    #print(path_filter[:,0,:])
                    if dumped_values.shape[1]>=self.n_samples:
                        print("Finished streamlines early, breaking!")
                        break
                if not dumped_values.shape[1]>=self.n_samples: #Still some samples remaining in the remainder
                    print("Streamlines remaining")
                    path_matrix_torch_new = torch.zeros((remainder+2, path_matrix_torch.shape[1], 3), dtype=torch.float32).cuda()
                    path_matrix_torch_new[0:2,...] = path_matrix_torch[-2:,...]
                    del path_matrix_torch
                    #For remainder
                    for i in range(remainder-1):
                        path_matrix_torch_new = propagate_topo_matrix_gpu_alt(path_matrix_torch_new, torch.tensor([i+2]).cuda(), x_gpu, Q_gpu, step_size_gpu)
                    path_matrix_torch_new, dumped_values, path_filter= batched_filter_gpu_end_alt(path_matrix_torch_new, dumped_values, i, dim_gpu, M, path_filter, remainder, current=True)
                    #print(path_matrix_torch_new, dumped_values, path_filter)
                else:
                    del path_matrix_torch

                distances, curvatures = compute_curv_and_dist_mat_gpu_alt(init_points[0,:,:], init_points[1,:,:], init_points[2,:,:],dumped_values[0,:,:],dumped_values[1,:,:],dumped_values[2,:,:])
                print(distances)
                print(curvatures)
                end_time = time.time()
                print(f"Time taken for {self.n_samples} calculations with N~{self.Q.shape}: {end_time - start_time:.2f} seconds")
                topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
                print(topology.shape)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            prof.export_chrome_trace("trace.json")  # Optional: Export trace to view it in Chrome's trace viewer
                

        else:
            torch.cuda.empty_cache()
            num_per_dim = round(self.n_samples ** (1/3))
            if num_per_dim ** 3 < self.n_samples:
                num_per_dim += 1
            self.n_samples = num_per_dim ** 3
            print("... > Computing Topo in Batches on a GPU!")
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



            path_matrix_torch, _, M, path_filter, _, x_gpu, Q_gpu, dim_gpu, step_size_gpu = initialize_streamline_grid_gpu_alt(self.center, self.x_vec_pt, self.y_vec_pt, self.dimensions, num_per_dim, self.step_size, self.GPU_batch_freq, self)

            #path_matrix_torch=torch.tensor(path_matrix, dtype=torch.float32).cuda() #Of shape (GPU_batch_freq, n_samples, 3)
            #path_filter=torch.tensor(path_filter, dtype=torch.float32).cuda() #Of shape (GPU_batch_freq, n_samples, 1)
            dumped_values=torch.tensor(np.empty((6,0,3)), dtype=torch.float32).cuda() #Of shape (6,0,3)

            max_num_batch = int((M+2-2)/(self.GPU_batch_freq-2)) #+2 since M+2 points need to be computed, and -2 since first two initial points are already computed, and propagations of 98 steps
            remainder = (M+2-2)%(self.GPU_batch_freq-2)
            #print(path_matrix_torch.shape)
            #print(dumped_values.shape)
            #print(path_filter.shape)
            #For streamline points not including remainder
            start_time = time.time()
            for i in range(max_num_batch):
                print(i)
                for j in range(self.GPU_batch_freq-2):
                    path_matrix_torch = propagate_topo_matrix_gpu_alt(path_matrix_torch, torch.tensor([j+1]).cuda(), x_gpu, Q_gpu, step_size_gpu)
                    if i==0 and j==0:
                        init_points = path_matrix_torch[0:3,...]
                print("filtering!")
                #print(path_matrix_torch[:,0,:])
                #print(path_filter[:,0,:])
                path_matrix_torch, dumped_values, path_filter, init_points = batched_filter_gpu_alt(path_matrix_torch, dumped_values, i, dim_gpu, M, path_filter, self.GPU_batch_freq, init_points, current=True)
                #print(path_matrix_torch[:,0,:])
                #print(path_filter[:,0,:])
                if dumped_values.shape[1]>=self.n_samples:
                    print("Finished streamlines early, breaking!")
                    break
            if not dumped_values.shape[1]>=self.n_samples: #Still some samples remaining in the remainder
                print("Streamlines remaining")
                path_matrix_torch_new = torch.zeros((remainder+2, path_matrix_torch.shape[1], 3), dtype=torch.float32).cuda()
                path_matrix_torch_new[0:2,...] = path_matrix_torch[-2:,...]
                del path_matrix_torch
                #For remainder
                for i in range(remainder-1):
                    path_matrix_torch_new = propagate_topo_matrix_gpu_alt(path_matrix_torch_new, torch.tensor([i+2]).cuda(), x_gpu, Q_gpu, step_size_gpu)
                path_matrix_torch_new, dumped_values, path_filter, init_points = batched_filter_gpu_end_alt(path_matrix_torch_new, dumped_values, i, dim_gpu, M, path_filter, remainder, init_points, current=True)
                #print(path_matrix_torch_new, dumped_values, path_filter)
            else:
                del path_matrix_torch
            print(init_points.shape)
            np.savetxt("dumped_values_init_alt.txt", dumped_values[0:3].cpu().numpy().transpose(1, 0, 2).reshape(dumped_values.shape[1], -1), fmt='%.6f')
            np.savetxt("dumped_values_final_alt.txt", dumped_values[3:6].cpu().numpy().transpose(1, 0, 2).reshape(dumped_values.shape[1], -1), fmt='%.6f')
            distances, curvatures = compute_curv_and_dist_mat_gpu(dumped_values[0,:,:], dumped_values[1,:,:], dumped_values[2,:,:],dumped_values[3,:,:],dumped_values[4,:,:],dumped_values[5,:,:])
            print(distances)
            print(curvatures)
            end_time = time.time()
            print(f"Time taken for {self.n_samples} calculations with N~{self.Q.shape}: {end_time - start_time:.2f} seconds")
            topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
            print(topology.shape)
        return topology