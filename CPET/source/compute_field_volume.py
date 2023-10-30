import numpy as np
import argparse
import json
from CPET.utils.parser import parse_pqr
from CPET.utils.calculator import compute_field_on_grid
from CPET.utils.writeoutput import write_field_to_file

def initialize_grid(center, x, y, density, dimensions, offset):
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
    transformation_matrix = np.column_stack([x_unit, y_unit, z_unit])
    # Create arrays of coordinates along each axis
    x_coords = np.linspace(-dimensions[0], dimensions[0], 2*density[0]+1)
    y_coords = np.linspace(-dimensions[1], dimensions[1], 2*density[1]+1)
    z_coords = np.linspace(-dimensions[2], dimensions[2], 2*density[2]+1)
    # Create a meshgrid of coordinates
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_coords, y_coords, z_coords)
    # Stack the meshgrid coordinates into a single array of shape (density[0], density[1], density[2], 3)
    local_coords = np.stack([x_mesh, y_mesh, z_mesh], axis=-1)
    return local_coords, transformation_matrix

parser = argparse.ArgumentParser(description='This script computes a field in a box volume with given density')
parser.add_argument('-i', help='Input file name')
parser.add_argument('-o', help='Output file name')
args = parser.parse_args()
with open(args.i, 'r') as file:
    input_data = json.load(file)
center = np.array(input_data["center"])
x = np.array(input_data["x"])
y = np.array(input_data["y"])
dimensions = np.array(input_data["dimensions"])
density = np.array(input_data["density"])
offset = input_data["offset"]
path_to_pqr = input_data["pqr_file"]
grid_points, transformation_matrix = initialize_grid(center, x, y, density, dimensions, offset)
x, Q = parse_pqr(path_to_pqr)
x = (x-center)@transformation_matrix
field_points = compute_field_on_grid(grid_points, x, Q)
write_field_to_file(grid_points, field_points, args.o)
