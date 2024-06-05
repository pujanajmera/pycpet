# compute_topol_grid_activefilter.py
from CPET.utils.gpu import (
    compute_curv_and_dist_gpu,
    propagate_topo_matrix_gpu,
    batched_filter_gpu,
    initialize_streamline_grid_gpu,
)
from CPET.utils.parser import parse_pqr
import torch
import numpy as np
import time


# @profile
def main():
    options = {
        "path_to_pqr": "/home/santiagovargas/dev/CPET-python/tests/test_files/test_large.pqr",
        "center": [104.785, 113.388, 117.966],
        "x": [105.785, 113.388, 117.966],
        "y": [104.785, 114.388, 117.966],
        "n_samples": 10000,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.001,
        "GPU_batch_freq": 100,
    }

    x, Q = parse_pqr(options["path_to_pqr"])

    center = np.array(options["center"])
    x_vec_pt = np.array(options["x"])
    y_vec_pt = np.array(options["y"])
    dimensions = np.array(options["dimensions"])
    step_size = options["step_size"]
    n_samples = options["n_samples"]
    GPU_batch_freq = options["GPU_batch_freq"]

    Q = torch.tensor(Q).cuda()
    (
        path_matrix,
        transformation_matrix,
        M,
        path_filter,
        random_max_samples,
    ) = initialize_streamline_grid_gpu(
        center, x_vec_pt, y_vec_pt, dimensions, n_samples, step_size
    )
    path_matrix_torch = torch.tensor(path_matrix).cuda()
    path_filter = torch.tensor(path_filter).cuda()
    dumped_values = torch.tensor(np.empty((6, 0, 3))).cuda()
    x = (x - center) @ np.linalg.inv(transformation_matrix)
    x = torch.tensor(x).cuda()
    j = 0
    start_time = time.time()

    for i in range(len(path_matrix)):
        if i % 100 == 0:
            print(i)

        if j == len(path_matrix) - 1:
            break
        path_matrix_torch = propagate_topo_matrix_gpu(
            path_matrix_torch, i, x, Q, step_size
        )
        if i % GPU_batch_freq == 0 and i > 5:
            path_matrix_torch, dumped_values, path_filter = batched_filter_gpu(
                path_matrix_torch,
                dumped_values,
                i,
                dimensions,
                M,
                path_filter,
                current=True,
            )
            # GPU_batch_freq *= 2
        j += 1
        torch.cuda.empty_cache()
        if dumped_values.shape[1] >= n_samples:
            break
    # path_matrix_torch, dumped_values, path_filter= batched_filter(path_matrix_torch, dumped_values, i,dimensions, M, path_filter, current=False)
    print(dumped_values[:, 0, :])
    distances, curvatures = compute_curv_and_dist_gpu(
        dumped_values[0, :, :],
        dumped_values[1, :, :],
        dumped_values[2, :, :],
        dumped_values[3, :, :],
        dumped_values[4, :, :],
        dumped_values[5, :, :],
    )
    end_time = time.time()
    print(
        f"Time taken for {options['n_samples']} calculations with N~{Q.shape}: {end_time - start_time:.2f} seconds"
    )
    topology = np.column_stack((distances.cpu().numpy(), curvatures.cpu().numpy()))
    np.savetxt("hist_cpet_mat.txt", topology)
    return topology


main()
