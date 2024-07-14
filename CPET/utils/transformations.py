import numpy as np


def get_transformation(center, x, y):
    # Convert lists to numpy arrays
    x = x - center  # Translate to origin
    y = y - center  # Translate to origin
    # Normalize the vectors
    x_unit = x / np.linalg.norm(x)
    y_unit = y / np.linalg.norm(y)
    # Calculate the z unit vector by taking the cross product of x and y
    z_unit = np.cross(x_unit, y_unit)
    z_unit = z_unit / np.linalg.norm(z_unit)
    # Recalculate the y unit vector
    y_unit = np.cross(z_unit, x_unit)
    y_unit = y_unit / np.linalg.norm(y_unit)
    transformation_matrix = np.column_stack([x_unit, y_unit, z_unit]).T
    return transformation_matrix


def split_and_filter(
    mat,
    cutoff=95,
    min_max=True,
    std_mean=False,
    log1=False,
    unlog1=False,
    cos_center_scaling=False,
    sparsify=False,
    sparse_factor=1,
):
    mag = np.sqrt(np.sum(mat**2, axis=3))

    arr_mean = np.mean(mag)
    arr_std = np.std(mag)
    arr_min = np.min(mag)
    arr_max = np.max(mag)

    if log1:
        x_sign = np.sign(mat)
        # getting absolute value of every element
        x_abs = np.abs(mat)
        # applying log1p
        x_log1p = np.log1p(x_abs)
        # getting sign back
        mat = np.multiply(x_log1p, x_sign)

    if unlog1:
        print("invert log operation")
        x_sign = np.sign(mat)
        # getting absolute value of every element
        x_abs = np.abs(mat)
        # applying log1p
        x_unlog1p = np.expm1(x_abs)
        # getting sign back
        mat = np.multiply(x_unlog1p, x_sign)

    if min_max:
        mat = (mat - arr_min) / (arr_max - arr_min + 10e-10)

    if std_mean:
        mat = (mat - arr_mean) / (arr_std)

    if cos_center_scaling:
        shape = mat.shape
        center_ind = np.array(
            [np.ceil(shape[0] // 2), np.ceil(shape[1] // 2), np.ceil(shape[2] // 2)]
        )
        scale_mat = np.zeros_like(mat)
        max_dist = np.sqrt(np.sum(center_ind) ** 2)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    scale_mat[i, j, k] = 1 + 5 * np.cos(
                        np.sqrt(np.sum((center_ind - np.array([i, j, k])) ** 2))
                        / max_dist
                        * np.pi
                        / 2
                    )
        multiply = np.multiply(mat, scale_mat)
        mat = multiply

    try:
        u = mat[0][:, :, :, 0].flatten()
        v = mat[0][:, :, :, 1].flatten()
        w = mat[0][:, :, :, 2].flatten()
    except:
        u = mat[:, :, :, 0].flatten()
        v = mat[:, :, :, 1].flatten()
        w = mat[:, :, :, 2].flatten()

    component_distro = [
        np.sqrt(u[ind] ** 2 + v[ind] ** 2 + w[ind] ** 2) for ind in range(len(u))
    ]
    cutoff = np.percentile(component_distro, cutoff)

    for ind, i in enumerate(component_distro):
        if i < cutoff:
            u[ind], v[ind], w[ind] = 0, 0, 0

    u = np.around(u, decimals=2)
    v = np.around(v, decimals=2)
    w = np.around(w, decimals=2)

    if sparsify:
        u_zeros = np.zeros_like(u)
        v_zeros = np.zeros_like(v)
        w_zeros = np.zeros_like(w)
        u_zeros[::sparse_factor] = u[::sparse_factor]
        v_zeros[::sparse_factor] = v[::sparse_factor]
        w_zeros[::sparse_factor] = w[::sparse_factor]
        u = u_zeros
        v = v_zeros
        w = w_zeros

    return u, v, w
