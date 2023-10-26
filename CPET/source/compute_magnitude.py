import numpy as np
import argparse
import json
from CPET.utils.parser import parse_pqr, filter_pqr
from CPET.utils.calculator import calculate_field_at_point
from CPET.utils.writeoutput import write_field_to_file
from CPET.utils.transformations import get_transformation


def main():
    seek_number = 31356  # id of atom of interest
    # path_to_pqr = input_data["pqr_file"]
    path_to_pqr = "../../tests/test_files/test_large.pqr"
    x, Q, atom_num_list = parse_pqr(path_to_pqr, ret_atom_names=True)
    # center = np.array([104.785, 113.388, 117.966])

    # get atom number of the atom we are interested in
    seek_index = atom_num_list.index(seek_number)
    center = x[seek_index]
    x_axis = np.array(np.array([1, 0, 0]))
    y_axis = np.array(np.array([0, 1, 0]))
    transformation_matrix = get_transformation(center, x_axis, y_axis)
    x = (x - center) @ np.linalg.inv(transformation_matrix)
    x, Q = filter_pqr(x, Q, np.array([0.0, 0.0, 0.0]), radius=2.0)
    mag = calculate_field_at_point(x, Q)
    print(mag)


main()
