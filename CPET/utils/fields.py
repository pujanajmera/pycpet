import numpy as np 


def save_numpy_as_dat(dict_meta_data, average_field, name):
    """
    Saves np array in original format from cpet output
    Takes:
        dict_meta_data: dictionary with meta data
        average_field: np array with average field
        name: name of file to save
    """

    first_line = dict_meta_data["first_line"]
    steps_x = dict_meta_data["steps_x"]
    steps_y = dict_meta_data["steps_y"]
    steps_z = dict_meta_data["steps_z"]
    step_size_x = dict_meta_data["step_size_x"]
    step_size_y = dict_meta_data["step_size_y"]
    step_size_z = dict_meta_data["step_size_z"]
    print(dict_meta_data)
    lines = [first_line]
    # add six lines starting with #
    lines = lines + ["#\n"] * 6

    # write as
    for i in range(steps_x):
        for j in range(steps_y):
            for k in range(steps_z):
                line = "{:.3f} {:.3f} {:.3f} {:.6f} {:.6f} {:.6f}\n".format(
                    step_size_x * (i - (steps_x - 1) / 2),
                    step_size_y * (j - (steps_y - 1) / 2),
                    step_size_z * (k - (steps_z - 1) / 2),
                    average_field[i, j, k, 0],
                    average_field[i, j, k, 1],
                    average_field[i, j, k, 2],
                )
                lines.append(line)

    with open(name, "w") as f:
        f.writelines(lines)


def average_fields(mat):
    """Average the fields in the matrix."""
    return np.mean(mat, axis=0)


def aug_all(mat, target, xy=True, z=False, mut=False):
    full_aug = []
    full_aug_target = []

    for ind, i in enumerate(mat):
        x_aug, y_aug = augment_mat_field(i, target[ind])
        [full_aug.append(j) for j in x_aug]
        [full_aug_target.append(j) for j in y_aug]

    return np.array(full_aug), np.array(full_aug_target)


def augment_mat_field(mat, target=None, xy=True, z=False):
    """
    Utility function to augment the matrix and target. Also works without a target
    Takes:
        mat: matrix to augment
        target: target to augment
        xy: boolean to augment in xy plane
        z: boolean to augment in z plane
    Returns:
        aug_mat: augmented matrix
        aug_target(optionally): augmented target
    """

    aug_target = []
    aug_mat = []

    if xy:
        x_flip = np.array(np.flip(mat, axis=0), dtype=float)
        y_flip = np.array(np.flip(mat, axis=1), dtype=float)
        xy_flip = np.array(np.flip(np.flip(mat, axis=1), axis=0), dtype=float)

        x_flip[:, :, :, 0] = -1 * x_flip[:, :, :, 0]
        y_flip[:, :, :, 1] = -1 * y_flip[:, :, :, 1]
        xy_flip[:, :, :, 0] = -1 * xy_flip[:, :, :, 0]
        xy_flip[:, :, :, 1] = -1 * xy_flip[:, :, :, 1]

        aug_mat.append(mat)
        aug_mat.append(x_flip)
        aug_mat.append(y_flip)
        aug_mat.append(xy_flip)
        if target is not None:
            aug_target.append(target)
            aug_target.append(target)
            aug_target.append(target)
            aug_target.append(target)

    if z:
        z_flip = np.array(np.flip(mat, axis=2), dtype=float)
        xz_flip = np.array(np.flip(np.flip(mat, axis=0), axis=2), dtype=float)
        yz_flip = np.array(np.flip(np.flip(mat, axis=1), axis=2), dtype=float)
        xyz_flip = np.array(
            np.flip(np.flip(np.flip(mat, axis=2), axis=1), axis=0), dtype=float
        )

        z_flip[:, :, :, 0] = -1 * z_flip[:, :, :, 0]
        xz_flip[:, :, :, 0] = -1 * xz_flip[:, :, :, 0]
        xz_flip[:, :, :, 2] = -1 * xz_flip[:, :, :, 2]
        yz_flip[:, :, :, 1] = -1 * yz_flip[:, :, :, 1]
        yz_flip[:, :, :, 2] = -1 * yz_flip[:, :, :, 2]
        xyz_flip[:, :, :, 0] = -1 * xyz_flip[:, :, :, 0]
        xyz_flip[:, :, :, 1] = -1 * xyz_flip[:, :, :, 1]
        xyz_flip[:, :, :, 2] = -1 * xyz_flip[:, :, :, 2]

        aug_mat.append(z_flip)
        aug_mat.append(xz_flip)
        aug_mat.append(yz_flip)
        aug_mat.append(xyz_flip)
        if target is not None:
            aug_target.append(target)
            aug_target.append(target)
            aug_target.append(target)
            aug_target.append(target)

    if target == None:
        return aug_mat
    else:
        return aug_mat, aug_target
