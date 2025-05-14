import numpy as np
import time
from multiprocessing import Pool
import torch
import logging


from CPET.utils.parallel import task, task_batch, task_base, task_complete_thread
from CPET.utils.calculator import (
    initialize_box_points_random,
    initialize_box_points_uniform,
    compute_field_on_grid,
    calculate_electric_field_c_shared_full_alt,
    compute_ESP_on_grid,
)
from CPET.utils.io import (
    parse_pdb,
    parse_pqr,
    get_atoms_for_axes,
    filter_radius_whole_residue,
    filter_residue,
    filter_in_box,
    calculate_center,
    filter_resnum,
    filter_resnum_andname,
    filter_IDs,
    default_options_initializer,
)
from CPET.utils.gpu import (
    propagate_topo_matrix_gpu,
    compute_curv_and_dist_mat_gpu,
    batched_filter_gpu,
    generate_path_filter_gpu,
)

class calculator:
    """Initialize the calculator object with the following parameters
    
    Parameters
    ----------
    options : dict                           
        Dictionary from input options.json file
    path_to_pdb : str, optional
        Path to the PDB file, by default None
    
    Attributes
    ----------
    self.profile : Bool
        If True, the calculator will profile the code (development, intended for GPU)
    self.path_to_pdb : str
        Path to the PDB file
    self.step_size : float
        Step size for the electric field calculation, relevant for the topology calculation or volume field calcs
    self.dimensions : list
        Dimensions of the box for the electric field/topology calculations
    self.n_samples : int
        Number of streamlines used for the topology calculation
    self.max_streamline_init : str
        Method for determining when streamlines end
    self.concur_slip : int
        Number of concurrent processes for the CPU-accelerated topology calculation
    self.GPU_batch_freq : int
        Number of batches for the GPU-accelerated topology calculation
    self.dtype : str
        Data type for the calculations (float32 or float64)
    """
    def __init__(self, options, path_to_pdb=None):
        # self.efield_calc = calculator(math_loc=math_loc)
        options = default_options_initializer(options) # Double in case calculator is called outside of CPET.py
        self.profile = options["profile"]
        self.path_to_pdb = path_to_pdb

        # Electric field grid parameters
        self.step_size = options["step_size"]
        self.dimensions = (
            np.array(options["dimensions"]) if "dimensions" in options.keys() else None
        )

        #Topology calculation parameters
        self.n_samples = options["n_samples"] if "n_samples" in options.keys() else None
        self.concur_slip = options["concur_slip"]
        self.GPU_batch_freq = options["GPU_batch_freq"]
        self.dtype = options["dtype"]
        self.max_streamline_init = (
            options["max_streamline_init"]
            if "max_streamline_init" in options.keys()
            else "true_rand"
        )

        self.write_transformed_pdb = options["write_transformed_pdb"]
        self.strip_filter = (
            options["strip_filter"] if "strip_filter" in options.keys() else False
        )

        # Be very careful with the box_shift option. The box needs to be centered at the origin and therefore, the code will shift protein in the opposite direction of the provided box vector
        self.box_shift = (
            options["box_shift"] if "box_shift" in options.keys() else [0, 0, 0]
        )

        if ".pqr" in self.path_to_pdb:
            (
                self.x,
                self.Q,
                self.atom_number,
                self.resids,
                self.residue_number,
                self.atom_type,
            ) = parse_pqr(self.path_to_pdb)
        else:
            (
                self.x,
                self.Q,
                self.atom_number,
                self.resids,
                self.residue_number,
                self.atom_type,
                self.chains,
            ) = parse_pdb(self.path_to_pdb, get_charges=True)

        # Make ID list that has all information besides coordinates and charges
        if hasattr(self, "chains"):
            self.ID = [
                (
                    self.atom_number[i],
                    self.atom_type[i],
                    self.resids[i],
                    self.residue_number[i],
                    self.chains[i],
                )
                for i in range(len(self.x))
            ]
        else:
            self.ID = [
                (
                    self.atom_number[i],
                    self.atom_type[i],
                    self.resids[i],
                    self.residue_number[i],
                )
                for i in range(len(self.x))
            ]

        print(self.chains)
        ##################### define center

        if type(options["center"]) == list:
            self.center = np.array(options["center"])

        elif type(options["center"]) == dict:
            method = options["center"]["method"]
            pos_considered = get_atoms_for_axes(
                self.x,
                self.atom_type,
                self.residue_number,
                self.chains,
                options,
                seltype="center",
            )
            self.center = calculate_center(pos_considered, method=method)
        else:
            raise ValueError("center must be a list or dict")

        ##################### define x axis

        # First, failsafe if x doesn't exist in options dict:
        if "x" not in options.keys():
            print(
                "No x specified, calculating in input file reference frame. Ignoring y..."
            )
            compute_y = False
        elif type(options["x"]) == list:
            self.x_vec_pt = np.array(options["x"])
            compute_y = True
        elif type(options["x"]) == dict:
            method = options["x"]["method"]
            pos_considered = get_atoms_for_axes(
                self.x,
                self.atom_type,
                self.residue_number,
                self.chains,
                options,
                seltype="x",
            )
            self.x_vec_pt = calculate_center(pos_considered, method=method)
            compute_y = True
        else:
            # Return an error:
            raise ValueError("x must be a list or dict")

        ##################### define y axis

        if compute_y == False:
            print("Not computing y since x is not specified")
        elif type(options["y"]) == list:
            self.y_vec_pt = np.array(options["y"])

        elif type(options["y"]) == dict:
            method = options["y"]["method"]
            pos_considered = get_atoms_for_axes(
                self.x,
                self.atom_type,
                self.residue_number,
                self.chains,
                options,
                seltype="y",
            )
            self.y_vec_pt = calculate_center(pos_considered, method=method)
        else:
            raise ValueError("Since you have provided x, y must be a list or dict")

        self.x_copy = self.x
        self.residue_number_copy = self.residue_number
        self.resids_copy = self.resids
        self.atom_number_copy = self.atom_number
        self.atom_type_copy = self.atom_type

        # Any sort of filtering related to atom identity information
        # NEED TO MAKE MORE ROBUST
        if "filter_IDs" in options.keys():
            self.x, self.Q, self.ID = filter_IDs(
                self.x, self.Q, self.ID, options["filter_IDs"]
            )

            if hasattr(self, "chains"):
                self.atom_number = [self.ID[i][0] for i in range(len(self.x))]
                self.atom_type = [self.ID[i][1] for i in range(len(self.x))]
                self.resids = [self.ID[i][2] for i in range(len(self.x))]
                self.residue_number = [self.ID[i][3] for i in range(len(self.x))]
                self.chains = [self.ID[i][4] for i in range(len(self.x))]
            else:
                self.atom_number = [self.ID[i][0] for i in range(len(self.x))]
                self.atom_type = [self.ID[i][1] for i in range(len(self.x))]
                self.resids = [self.ID[i][2] for i in range(len(self.x))]
                self.residue_number = [self.ID[i][3] for i in range(len(self.x))]

        else:
            if "filter_resids" in options.keys():
                # print("filtering residues: {}".format(options["filter_resids"]))
                (
                    self.x,
                    self.Q,
                    self.residue_number,
                    self.resids,
                    self.atom_number,
                    self.atom_type,
                ) = filter_residue(
                    self.x,
                    self.Q,
                    self.residue_number,
                    self.resids,
                    self.atom_number,
                    self.atom_type,
                    filter_list=options["filter_resids"],
                )

            if "filter_resnum" in options.keys():
                # print("filtering residues: {}".format(options["filter_resids"]))
                self.x, self.Q, self.residue_number, self.resids = filter_resnum(
                    self.x,
                    self.Q,
                    self.residue_number,
                    self.resids,
                    filter_list=options["filter_resnum"],
                )

            if "filter_resnum_andname" in options.keys():
                # print("filtering residues: {}".format(options["filter_resids"]))
                (
                    self.x,
                    self.Q,
                    self.residue_number,
                    self.resids,
                    self.atom_number,
                    self.atom_type,
                ) = filter_resnum_andname(
                    self.x,
                    self.Q,
                    self.residue_number,
                    self.resids,
                    self.atom_number,
                    self.atom_type,
                    filter_list=options["filter_resnum_andname"],
                )

        if "filter_radius" in options.keys():
            print("filtering by radius: {} Ang".format(options["filter_radius"]))

            r = np.linalg.norm(self.x, axis=1)
            # print("r {}".format(r))

            # Default is whole residue-inclusive filtering to ensure proper topology convergence
            self.x, self.Q = filter_radius_whole_residue(
                x=self.x,
                Q=self.Q,
                resids=self.resids,
                resnums=self.residue_number,
                center=self.center,
                radius=float(options["filter_radius"]),
            )

            # print("center {}".format(self.center))
            r = np.linalg.norm(self.x, axis=1)
            # print("r {}".format(r))

        if "filter_in_box" in options.keys():
            if bool(options["filter_in_box"]):
                # print("filtering charges in sampling box")
                self.x, self.Q = filter_in_box(
                    x=self.x, Q=self.Q, center=self.center, dimensions=self.dimensions
                )

        assert "CPET_method" in options.keys(), "CPET_method must be specified"

        if (
            options["CPET_method"] == "volume" or options["CPET_method"] == "volume_ESP"
        ) and hasattr(self, "y_vec_pt"):
            N_cr = 2 * self.dimensions / self.step_size
            N_cr = [int(N_cr[0]), int(N_cr[1]), int(N_cr[2])]
            (self.mesh, self.transformation_matrix) = initialize_box_points_uniform(
                center=self.center,
                x=self.x_vec_pt,
                y=self.y_vec_pt,
                N_cr=N_cr,
                dimensions=self.dimensions,
                dtype=self.dtype,
                inclusive=True,
            )
        elif (
            options["CPET_method"] == "topo" or options["CPET_method"] == "topo_GPU"
        ) and hasattr(self, "y_vec_pt"):
            self.max_steps = round(2 * np.linalg.norm(self.dimensions) / self.step_size)
            if options["initializer"] == "random":
                (
                    self.random_start_points,
                    self.random_max_samples,
                    self.transformation_matrix,
                    self.max_streamline_len,
                ) = initialize_box_points_random(
                    self.center,
                    self.x_vec_pt,
                    self.y_vec_pt,
                    self.dimensions,
                    self.n_samples,
                    dtype=self.dtype,
                    max_steps=self.max_steps,
                )
            elif options["initializer"] == "uniform":
                num_per_dim = round(self.n_samples ** (1 / 3))
                if num_per_dim**3 < self.n_samples:
                    num_per_dim += 1
                self.n_samples = num_per_dim**3
                # print("num_per_dim: ", num_per_dim)
                grid_density = 2 * self.dimensions / (num_per_dim + 1)
                print("grid_density: ", grid_density)
                seed = None
                if self.max_streamline_init == "fixed_rand":
                    print("Fixing max steps with Random seed 42")
                    seed = 42
                (
                    self.random_start_points,
                    self.random_max_samples,
                    self.transformation_matrix,
                ) = initialize_box_points_uniform(
                    center=self.center,
                    x=self.x_vec_pt,
                    y=self.y_vec_pt,
                    dimensions=self.dimensions,
                    N_cr=[num_per_dim, num_per_dim, num_per_dim],
                    dtype=self.dtype,
                    max_steps=self.max_steps,
                    ret_rand_max=True,
                    inclusive=False,
                    seed=seed,
                )
                # convert mesh to list of x, y, z points
                # print(self.random_start_points)
                self.random_start_points = self.random_start_points.reshape(-1, 3)
                self.n_samples = len(self.random_start_points)
                # print("random start points")
                # print(self.random_start_points)
                print("start point shape: ", str(self.random_start_points.shape))
        elif (
            options["CPET_method"] == "point_field"
            or options["CPET_method"] == "point_mag"
            or options["CPET_method"] == "box_check"
        ) and hasattr(self, "y_vec_pt"):
            (
                _,
                _,
                self.transformation_matrix,
            ) = initialize_box_points_uniform(
                center=self.center,
                x=self.x_vec_pt,
                y=self.y_vec_pt,
                dimensions=[0, 0, 0],
                N_cr=[0, 0, 0],
                dtype=self.dtype,
                max_steps=0,
                ret_rand_max=True,
                inclusive=False,
            )

        if hasattr(self, "y_vec_pt"):
            print("Rotating coordinates")
            self.x = (self.x - self.center) @ np.linalg.inv(self.transformation_matrix)
        else:
            print("Not rotating coordinates since no y-vector is provided")
            self.x = self.x - self.center
        """
        #Debug version
        self.x_copy = self.x
        self.residue_number_copy = self.residue_number
        self.resids_copy = self.resids
        self.atom_number_copy = self.atom_number
        self.atom_type_copy = self.atom_type
        """
        if self.write_transformed_pdb == True:
            print("Writing transformed pdb file, ignoring chains")
            if self.strip_filter == True:
                print("Stripping filtered residues for transformed pdb file")
                self.x_copy = self.x
                self.residue_number_copy = self.residue_number
                self.resids_copy = self.resids
                self.atom_number_copy = self.atom_number
                self.atom_type_copy = self.atom_type
            self.x_copy = (self.x_copy - self.center) @ np.linalg.inv(
                self.transformation_matrix
            )
            chain_id = "A"
            with open(f"transform_{path_to_pdb.split('/')[-1][:-4]}.pdb", "w") as f:
                for i in range(len(self.x_copy)):
                    atom_number = int(float(self.atom_number_copy[i]))
                    atom_name = self.atom_type_copy[i]  # The atom name (like 'CA', 'O')
                    residue_name = self.resids_copy[i]  # Residue name (like 'ALA')
                    residue_number = int(float(self.residue_number_copy[i]))
                    x, y, z = self.x_copy[i]

                    # PDB format: ATOM or HETATM, atom number, atom name, residue name, chain (default 'A'),
                    # residue number, coordinates (x, y, z), occupancy (default 1.00), temperature factor (default 0.00)
                    f.write(
                        f"ATOM  {atom_number:>5d} {atom_name:<4} {residue_name:>3} {chain_id:>1}{residue_number:>4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           \n"
                    )

        # print(self.center)
        # print(self.x_vec_pt)
        # print(self.y_vec_pt)
        if self.box_shift != [0, 0, 0]:
            print("Shifting box by: ", self.box_shift)
            self.x = self.x - np.array(self.box_shift)

        if self.dtype == "float32":
            self.x = self.x.astype(np.float32)
            self.Q = self.Q.astype(np.float32)
            self.center = self.center.astype(np.float32)
            if self.dimensions is not None:
                self.dimensions = self.dimensions.astype(np.float32)

        print("... > Initialized Calculator!")

    def compute_point_mag(self):
        """Compute the electric field magnitude at a point defined by the center and the charges Q at positions x.
        
        Returns
        -------
        point_mag : np.ndarray
            The computed electric field magnitude at the point
        """
        print("... > Computing Point Field (Magnitude)!")
        print(f"Number of charges: {len(self.Q)}")
        print("point: {}".format(self.center))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        print("First few lines of x: {}".format(self.x[:5]))
        start_time = time.time()
        # Since x and Q are already rotated and translated, need to supply 0 vector as center
        point_mag = np.norm(
            calculate_electric_field_c_shared_full_alt(np.array([0, 0, 0]), self.x, self.Q)
        )
        end_time = time.time()
        print(f"{end_time - start_time:.2f}")
        return point_mag

    def compute_point_field(self):
        """Compute the electric field at a point defined by the center and the charges Q at positions x.

        Returns
        -------
        point_field : np.ndarray
            The computed electric field at the point
        """
        print("... > Computing Point Field (Vector)!")
        print(f"Number of charges: {len(self.Q)}")
        print("point: {}".format(self.center))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        print("First few lines of x: {}".format(self.x[:5]))
        start_time = time.time()
        # Since x and Q are already rotated and translated, need to supply 0 vector as center
        point_field = calculate_electric_field_c_shared_full_alt(
            np.array([0, 0, 0]), self.x, self.Q
        )
        end_time = time.time()
        print(f"{end_time - start_time}")
        return point_field

    def compute_box(self):
        """Compute the electric field on a grid defined by the mesh and the charges Q at positions x.
        
        Returns
        -------
        field_box : np.ndarray
            The computed electric field on the grid
        mesh.shape : tuple
            The shape of the mesh grid
        """
        print("... > Computing Box Field!")
        print(f"Number of charges: {len(self.Q)}")
        print("mesh shape: {}".format(self.mesh.shape))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        print("First few lines of x: {}".format(self.x[:5]))
        print("Transformation matrix: {}".format(self.transformation_matrix))
        field_box = compute_field_on_grid(self.mesh, self.x, self.Q)
        return field_box, self.mesh.shape

    def compute_box_ESP(self):
        """Compute the electrostatic potential on a grid defined by the mesh and the charges Q at positions x.
        
        Returns
        -------
        esp_box : np.ndarray
            The computed electrostatic potential on the grid
        mesh.shape : tuple
            The shape of the mesh grid
        """
        print("... > Computing Box ESP!")
        print(f"Number of charges: {len(self.Q)}")
        print("mesh shape: {}".format(self.mesh.shape))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        print("First few lines of x: {}".format(self.x[:5]))
        print("Transformation matrix: {}".format(self.transformation_matrix))
        print("Center: {}".format(self.center))
        esp_box = compute_ESP_on_grid(self.mesh, self.x, self.Q)
        return esp_box, self.mesh.shape


    def compute_topo_base(self):
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        # print("starting pooling")
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            result = pool.starmap(task_base, args)
            # print(raw)
            hist = []
            for result in result.get():
                hist.append(result)
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
        # print("starting pooling")
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            # raw = pool.starmap(task, args)

            result = pool.starmap_async(task, args)
            dist = []
            curve = []
            for result in result.get():
                dist.append(result[0])
                curve.append(result[1])

            hist = np.column_stack((dist, curve))
        end_time = time.time()
        self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist

    def compute_topo_single(self):
        print("... > Computing Topo(baseline)!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        dist_list, curve_list, init_points_list, final_points_list = [], [], [], []
        endtype_list = []
        for i, n_iter in zip(self.random_start_points, self.random_max_samples):

            dist, curve, init_points, final_points, endtype = task_base(
                i, n_iter, self.x, self.Q, self.step_size, self.dimensions
            )
            dist_list.append(dist)
            curve_list.append(curve)
            init_points_list.append(init_points)
            final_points_list.append(final_points)
            endtype_list.append(endtype)
            # print(dist, curve)
        hist = np.column_stack((dist_list, curve_list))
        end_time = time.time()
        self.hist = hist
        init_points_list = np.array(init_points_list)  # Shape (N, 3, 3)
        final_points_list = np.array(final_points_list)  # Shape (N, 3, 3)
        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        # For testing purposes
        np.savetxt(
            "topology_base.txt",
            hist,
            fmt="%.6f",
        )

        np.savetxt(
            "dumped_values_init_base.txt",
            init_points_list.reshape(init_points_list.shape[0], -1),
            fmt="%.6f",
        )
        np.savetxt(
            "dumped_values_final_base.txt",
            final_points_list.reshape(final_points_list.shape[0], -1),
            fmt="%.6f",
        )

        np.savetxt(
            "endtype_base.txt",
            endtype_list,
            fmt="%s",
        )
        return hist

    def compute_topo_complete_c_shared(self):
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        print(f"Start point shape: {self.random_start_points.shape}")
        start_time = time.time()
        # print("starting pooling")
        # print("random start points")
        # print(self.random_max_samples)
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            # raw = pool.starmap(task, args)

            result = pool.starmap_async(task_complete_thread, args)
            dist = []
            curve = []
            for result in result.get():
                dist.append(result[0])
                curve.append(result[1])

            hist = np.column_stack((dist, curve))

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
        # print(self.random_start_points_batched)
        # print(self.random_max_samples_batched)
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(
                    self.random_start_points_batched, self.random_max_samples_batched
                )
            ]
            raw = pool.starmap(task_batch, args)
            raw_flat = [item for sublist in raw for item in sublist]
            dist = []
            curve = []
            for result in raw_flat:
                dist.append(result[0])
                curve.append(result[1])
            hist = np.column_stack((dist, curve))

        end_time = time.time()
        # self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist

    def compute_topo_GPU_batch_filter(self):
        """Compute the topology using GPU with batch-filtering
        
        Returns
        -------
        hist : np.ndarray
            The computed topology data
        """

        print("... > Computing Topo in Batches on a GPU!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")

        Q_gpu = torch.tensor(self.Q, dtype=torch.float32).cuda()
        Q_gpu = Q_gpu.unsqueeze(0)
        x_gpu = torch.tensor(self.x, dtype=torch.float32).cuda()
        dim_gpu = torch.tensor(self.dimensions, dtype=torch.float32).cuda()
        step_size_gpu = torch.tensor([self.step_size], dtype=torch.float32).cuda()
        random_max_samples = torch.tensor(self.random_max_samples).cuda()

        M = self.max_steps
        max_num_batch = (
            int((M + 2 - self.GPU_batch_freq) / (self.GPU_batch_freq - 2)) + 1
        )
        remainder = (M + 2 - self.GPU_batch_freq) % (
            self.GPU_batch_freq - 2
        )  # Number of remaining propagations
        path_matrix_torch = torch.tensor(
            np.zeros((self.GPU_batch_freq, self.n_samples, 3)), dtype=torch.float32
        ).cuda()
        path_matrix_torch[0] = torch.tensor(self.random_start_points)
        path_matrix_torch = propagate_topo_matrix_gpu(
            path_matrix_torch,
            torch.tensor([0]).cuda(),
            x_gpu,
            Q_gpu,
            step_size_gpu,
        )
        # Using random_max_samples-1 to convert from max samples to indices
        path_filter = generate_path_filter_gpu(
            random_max_samples, torch.tensor([M + 2], dtype=torch.int64).cuda()
        )

        # Need to augment random_max_samples for smaller streamlines than the batching frequency
        if M + 2 < self.GPU_batch_freq:
            path_filter_temp = torch.ones(
                (self.GPU_batch_freq, self.n_samples, 1), dtype=torch.bool
            ).cuda()
            path_filter_temp[0 : M + 2] = path_filter
            path_filter = path_filter_temp

        path_filter = torch.tensor(path_filter, dtype=torch.bool).cuda()
        print(path_matrix_torch.shape)
        print(path_filter.shape)
        print(M + 2)

        dumped_values = torch.tensor(np.empty((6, 0, 3)), dtype=torch.float32).cuda()

        j = 0
        start_time = time.time()
        for i in range(max_num_batch):

            for j in range(self.GPU_batch_freq - 2):
                path_matrix_torch = propagate_topo_matrix_gpu(
                    path_matrix_torch,
                    torch.tensor([j + 1]).cuda(),
                    x_gpu,
                    Q_gpu,
                    step_size_gpu,
                )
                if i == 0 and j == 0:
                    init_points = path_matrix_torch[0:3, ...]

            # print("filtering!")
            (
                path_matrix_torch,
                dumped_values,
                path_filter,
                init_points,
            ) = batched_filter_gpu(
                path_matrix=path_matrix_torch,
                dumped_values=dumped_values,
                i=i,
                dimensions=dim_gpu,
                path_filter=path_filter,
                init_points=init_points,
                GPU_batch_freq=self.GPU_batch_freq,
                dtype_str=self.dtype,
            )
            print(dumped_values.shape[1])
            if dumped_values.shape[1] >= self.n_samples:
                print("Finished streamlines early, breaking!")
                break
        print(
            f"Checking dumped values ({dumped_values.shape[1]}) vs number of samples ({self.n_samples})"
        )
        if (
            dumped_values.shape[1] < self.n_samples
        ):  # Still some samples remaining in the remainder
            print("Streamlines remaining")
            print(remainder)
            print(path_matrix_torch.shape)
            path_matrix_torch_new = torch.zeros(
                (remainder + 2, path_matrix_torch.shape[1], 3),
                dtype=torch.float32,
            ).cuda()
            path_matrix_torch_new[0:2, ...] = path_matrix_torch[-2:, ...]
            del path_matrix_torch
            # For remainder
            for i in range(remainder - 1):
                path_matrix_torch_new = propagate_topo_matrix_gpu(
                    path_matrix_torch_new,
                    torch.tensor([i + 2]).cuda(),
                    x_gpu,
                    Q_gpu,
                    step_size_gpu,
                )
            (
                path_matrix_torch_new,
                dumped_values,
                path_filter,
                init_points,
            ) = batched_filter_gpu(
                path_matrix=path_matrix_torch_new,
                dumped_values=dumped_values,
                i=i,
                dimensions=dim_gpu,
                path_filter=path_filter,
                init_points=init_points,
                GPU_batch_freq=remainder,
                dtype_str=self.dtype,
            )
            # print(path_matrix_torch_new, dumped_values, path_filter)
        else:
            del path_matrix_torch

        print(dumped_values.shape)
        np.savetxt(
            "dumped_values_init.txt",
            dumped_values[0:3]
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
            .reshape(dumped_values.shape[1], -1),
            fmt="%.6f",
        )
        np.savetxt(
            "dumped_values_final.txt",
            dumped_values[3:6]
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
            .reshape(dumped_values.shape[1], -1),
            fmt="%.6f",
        )
        distances, curvatures = compute_curv_and_dist_mat_gpu(
            dumped_values[0, :, :],
            dumped_values[1, :, :],
            dumped_values[2, :, :],
            dumped_values[3, :, :],
            dumped_values[4, :, :],
            dumped_values[5, :, :],
        )
        end_time = time.time()
        print(
            f"Time taken for {self.n_samples} calculations with N~{self.Q.shape}: {end_time - start_time:.2f} seconds"
        )
        topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
        print(topology.shape)
        # For dev testing
        np.savetxt(
            "topology_GPU.txt",
            topology,
            fmt="%.6f",
        )
        return topology
