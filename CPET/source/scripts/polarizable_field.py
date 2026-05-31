from CPET.source.CPET import CPET
from CPET.utils import intro_citation

import numpy as np

"""
Section required for fortran binding imports
"""

"""
This script is for polarizable force field calculations of electric field.

Current status: Skeleton
"""


def parse_coordinates():
    """
    Parses the coordinate and parameter files to extract the coordinates, charges, dipoles, and quadrupoles
    """
    return 0, 0, 0, 0


def parse_topology():
    """
    Parses the topology file to extract the rotation matrices
    """
    return 0


def rotate_dipoles_quadrupoles(r, d, t):
    """
    Rotates dipoles and quadrupoles to align with the field
    Takes:
        r: rotation matrix of each atom of shape (N, 3, 3)
        d: dipole moments of the polarizable atoms of shape (N, 3)
        t: quadrupole moments of the polarizable atoms of shape (N, 3, 3)
    Returns:
        d: rotated dipole moments
        t: rotated quadrupole moments
    """
    d = np.einsum("nij,nj->ni", r, d)
    t = np.einsum("nij,njk->nik", r, t)

    return d, t


def rotate_to_box_reference(x, d, t):
    """
    Rotates coordinates, dipoles, and quadrupoles to align with the box reference frame
    """

    return x, d, t


def compute_volume_field():
    """
    Computes volume field in same format as typical cpet methods
    """
    return 0


def save_field(field, output_path):
    """
    Saves the computed field to the output path
    """
    return 0


def main():
    parameter_path = (
        "path/to/parameters"  # Includes permanent charges, dipoles, and quadrupoles
    )
    coordinate_path = "path/to/coordinates"  # Includes connectivity information
    dipole_path = "path/to/dipoles"  # Includes induced dipoles

    x, q, d, t = parse_coordinates(coordinate_path, parameter_path)
    r = parse_topology(coordinate_path, parameter_path)
    d, t = rotate_dipoles_quadrupoles(r, d, t)
    x, d, t = rotate_to_box_reference(x, d, t)
    field = compute_volume_field(x, q, d, t)
    save_field(field, "path/to/output")

    return 0


if __name__ == "__main__":
    main()
