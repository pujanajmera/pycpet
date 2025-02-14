import numpy as np
from glob import glob
import warnings
import math
# from chimerax.core.commands import run

"""
Overall script for all visualization functions and visualization-affiliates.

Current functions:
- visualize_fields: Visualize electric fields in proteins by creating a bild file
for Chimera/ChimeraX input

To-add:
- Time-dependent animations
- Clustering visualizations
- High quality distance matrix plotting (?)
"""

def visualize_esp(path_to_pdb, path_to_esp, outputpath, options):
    """
    Visualize electrostatic potential in proteins by creating a bild file
    for Chimera/ChimeraX input
    Takes:
        path_to_pdb: Path to the PDB file of the protein
        path_to_esp: Path to the corresponding ESP file
        options: Options dictionary
    """
    
    if "visualization" in options.keys():
        cutoff = options["visualization"]["cutoff"] if "cutoff" in options["visualization"] else 0
        sparsify_factor = options["visualization"]["sparsify_factor"] if "sparsify_factor" in options["visualization"] else 1
    else:
        cutoff = 0
        sparsify_factor = 1

    name = path_to_pdb.split("/")[-1].split(".")[0]

    bild_path = outputpath + "/" + name

    # Generate bild file
    (
        sample_density_array,
        volume_box_array,
        center_array,
        basis_matrix_array,
        field_array,
    ) = process_field_file(path_to_esp)
    transformed_field_array = transform_field(
        field_array, center_array, basis_matrix_array
        )
    tip_to_tail_vectors = generate_tip_tail_vectors(
        transformed_field_array.copy(), sample_density_array, volume_box_array
    )
    generate_bild_file(
        tip_to_tail_vectors, transformed_field_array, cutoff, sparsify_factor, bild_path, path_to_efield
    )
    return "Bild file saved for {}".format(name)

def visualize_field(path_to_pdb, path_to_efield, outputpath, options, display=False):
    """
    Visualize electric fields in proteins by creating a bild file
    for Chimera/ChimeraX input, with optional display in ChimeraX
    Takes:
        path_to_pdb: Path to the PDB file of the protein
        path_to_field: Path to the corresponding field file
        options: Options dictionary
        display: Boolean to display the fields in ChimeraX
    """

    # Extract key options

    if "visualization" in options.keys():
        cutoff = options["visualization"]["cutoff"] if "cutoff" in options["visualization"] else 0
        sparsify_factor = options["visualization"]["sparsify_factor"] if "sparsify_factor" in options["visualization"] else 1
    else:
        cutoff = 0
        sparsify_factor = 1

    name = path_to_pdb.split("/")[-1].split(".")[0]

    bild_path = outputpath + "/" + name

    # Generate bild file
    (
        sample_density_array,
        volume_box_array,
        center_array,
        basis_matrix_array,
        field_array,
    ) = process_field_file(path_to_efield)
    transformed_field_array = transform_field(
        field_array, center_array, basis_matrix_array
        )
    tip_to_tail_vectors = generate_tip_tail_vectors(
        transformed_field_array.copy(), sample_density_array, volume_box_array
    )
    generate_bild_file(
        tip_to_tail_vectors, transformed_field_array, cutoff, sparsify_factor, bild_path, path_to_efield
    )
    return "Bild file saved for {}".format(name)


def process_field_file(file_path):
    """
    This function processes all data from the field file
    Inputs:
        file_path: Path to the field file
    Outputs:
        sample_density: Sample density
        volume_box: Volume of the box in Angstroms
        center: Center of the box in Angstroms
        basis_matrix: Basis matrix used for rotation
        field: Electric field values
    """
    # Initialize variables
    sample_density = []
    volume_box = []
    center = []
    basis_matrix = []
    field = []

    # Read the file
    reading_basis_matrix = False
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#Sample Density"):
                parts = line.split(";")
                try:
                    sample_density = [float(x) for x in parts[0].split()[2:5]]
                except:
                    raise ValueError("Sample density line not formatted correctly")
                try:
                    volume_box = [float(x) for x in parts[1].split()[2:5]]
                except:
                    raise ValueError(
                        "Volume box part of density line not formatted correctly"
                    )
            elif line.startswith("#Center"):
                try:
                    center = [float(x) for x in line.split()[1:4]]
                except:
                    raise ValueError("Center line not formatted correctly")
            elif line.startswith("#Basis Matrix"):
                reading_basis_matrix = True
            elif reading_basis_matrix and line.startswith("#"):
                try:
                    basis_matrix.append([float(x) for x in line[1:].split()[0:3]])
                except:
                    raise ValueError(
                        f"Basis matrix line not formatted correctly; matrix list of length {len(basis_matrix)}"
                    )
                if len(basis_matrix) == 3:
                    reading_basis_matrix = False
            elif not line.startswith("#"):
                try:
                    field.append([float(x) for x in line.split()])
                except:
                    raise ValueError(
                        f"Field line not formatted correctly; field list of length {len(field)}"
                    )
    if not sample_density:
        warnings.warn("Sample density not found, ignoring for checking")
    if not field:
        raise ValueError("No field data found at all, exiting...")
    if not center or not basis_matrix:
        warnings.warn(
            "Center or basis matrix not found, no transformation will be made"
        )
    if sample_density:
        check_field(np.array(field), np.array(sample_density))
    # print(np.array(basis_matrix))
    return (
        np.array(sample_density),
        np.array(volume_box),
        np.array(center),
        np.array(basis_matrix),
        np.array(field),
    )


def check_field(field_array, sample_density_array):
    """
    This function checks if the field provided matches the sample density
    Inputs:
        field_array: Field array
        sample_density_array: Sample density array
    Outputs:
        None
    """
    # print(sample_density_array)
    # print(field_array.shape)
    if field_array.shape[0] != np.product(sample_density_array, axis=0):
        raise ValueError(
            f"Field provided does not match sample density, field of shape {field_array.shape[0]} does not match expected sample amount of {np.product(sample_density_array, axis=0)}"
        )
    else:
        print("Field matches sample density, check passed, continuing...")


def transform_field(field_array, center_array, basis_matrix_array):
    """
    This function transforms the field to the new basis and new center
    Inputs:
        field_array: Field array
        center_array: Center array
        basis_matrix_array: Basis matrix array
    Outputs:
        transformed_field_array: Transformed field array
    """
    field_coords = field_array[:, 0:3]
    field_vecs = field_array[:, 3:6]
    # print(field_coords, field_vecs, basis_matrix_array)
    # Need to transform and recenter field coords, but only transform field vecs
    transformed_field_coords = field_coords @ basis_matrix_array.T
    transformed_field_coords = transformed_field_coords + center_array
    transformed_field_vecs = np.matmul(field_vecs, basis_matrix_array.T)
    transformed_field_array = np.concatenate(
        (transformed_field_coords, transformed_field_vecs), axis=1
    )
    return transformed_field_array


def generate_tip_tail_vectors(field_array, sample_density_array, volume_box_array):
    """
    This function generates the tip and tail vectors for the field
    Inputs:
        field_array: Field array (transformed or not) Shape (N, 6)
        sample_density_array: Sample density array Shape (3,)
        volume_box_array: Volume box array
    Outputs:
        tip_to_tail_vectors: Tip to tail vectors
    """

    max_vector_length = 0.9 * np.min(2*volume_box_array / sample_density_array)

    """
    if sample_density_array!=[] and volume_box_array!=[]:
        max_vector_length = 0.9*np.min(volume_box_array/sample_density_array)
    else:
        max_vector_length = 0.25
    """

    # Normalize the vectors in field_array
    field_array[:, 3:6] = (
        field_array[:, 3:6]
        / np.linalg.norm(field_array[:, 3:6], axis=1)[:, None]
        * max_vector_length
    )
    tail_vecs = field_array[:, 0:3] - field_array[:, 3:6] / 2
    tip_vecs = field_array[:, 0:3] + field_array[:, 3:6] / 2

    tip_to_tail_vectors = np.concatenate((tail_vecs, tip_vecs), axis=1)

    return tip_to_tail_vectors


def fac(num, f):
    """
    This function returns the factor of num which is closest to f
    Inputs:
        num
        f
    """
    fac_list = []
    for i in range(1, num + 1):
        if num % i == 0:
            fac_list.append(i)

    diff = num
    fac = 1
    for i in fac_list:
        if abs(f - i) <= diff:
            fac = i
            diff = abs(f - i)
    return fac


def sparsify_vec_field(
    vectors, sparsify_factor, nx, ny, nz, dim1, dim2, dim3, printflag=0
):
    """
    This function generates the sparsified 3D field vectors and corresponding colors
    Inputs:
        vectors: vectors in the vector field
        sparsify_factor
        nx: number of vectors in x direction
        ny: number of vectors in y direction
        nz: number of vectors in z direction
        printflag: flag to print number of points
    """
    # sparsify_factor=3
    # print(nx)
    # print(ny)
    # print(nz)

    max_dim = max(dim1, dim2, dim3)

    d1 = max_dim / dim1
    d2 = max_dim / dim2
    d3 = max_dim / dim3

    # sx = nx // fac(nx, dim1*sparsify_factor) #=3
    # sy = ny // fac(ny, dim2*sparsify_factor) #=7
    # sz = nz // fac(nz, dim3*sparsify_factor) #=7

    sx = fac(nx, d1 * sparsify_factor)  # =7
    sy = fac(ny, d2 * sparsify_factor)  # =3
    sz = fac(nz, d3 * sparsify_factor)  # =3

    # print("S", sx, sy, sz)
    indices = [i for i in range(len(vectors))]

    # sparsify along x direction
    indices_zs = indices[::sz]
    new_nz = nz / sz  # =7

    indices_yzs = []
    for i in range(len(indices_zs)):
        if (i // new_nz) % sy == 0:  # if (i // 7 ) % 7 == 0
            indices_yzs.append(indices_zs[i])
    new_ny = ny / sy  # =3
    # new_ny = ny/7
    # print(indices_xys)
    # print("new ny", new_ny)
    indices_xyzs = []
    for i in range(len(indices_yzs)):
        if (i // (new_ny * new_nz)) % sx == 0:  # if (i // 7*7) % 3 == 0
            indices_xyzs.append(indices_yzs[i])
    new_nx = nx / sx

    if printflag == 1:
        print("Original field has ", nx, " X ", ny, " X ", nz, " points.")
        print(
            "Sparsified field has ",
            int(new_nx),
            " X ",
            int(new_ny),
            " X ",
            int(new_nz),
            " points before filtering.",
        )

    vectors_s = []
    for i in indices_xyzs:
        vectors_s.append(vectors[i])
    vectors_s_np = np.array(vectors_s)
    # print(len(vectors_s_np) )
    return vectors_s_np


def scale_tip_to_tail_vectors(
    tip_to_tail_vectors, field_mags, sparsify_factor, min_dim
):
    """
    This function scales the tip to tail vectors based on their field magnitude
    Inputs:
        tip_to_tail_vectors: Tip to tail vectors
        field_mags: Field magnitudes to scale vectors
        sparsify_factor: for overall scale-up
        min_dim: the minimum of the three dimensions of the box, used for scaling
    """
    scaled_tip_to_tail_vectors = []
    for i in range(len(tip_to_tail_vectors)):

        # use field mag as scaling factor
        m = 5 * sparsify_factor * field_mags[i] * min_dim

        xA = tip_to_tail_vectors[i][0]
        yA = tip_to_tail_vectors[i][1]
        zA = tip_to_tail_vectors[i][2]

        xB = tip_to_tail_vectors[i][3]
        yB = tip_to_tail_vectors[i][4]
        zB = tip_to_tail_vectors[i][5]

        # find midpt of tip to tail vector
        xO = 0.5 * (xA + xB)
        yO = 0.5 * (yA + yB)
        zO = 0.5 * (zA + zB)

        # find new scaled tip
        xC = m * xB + (1 - m) * xO
        yC = m * yB + (1 - m) * yO
        zC = m * zB + (1 - m) * zO

        # find new scaled tail
        xD = 2 * xO - xC
        yD = 2 * yO - yC
        zD = 2 * zO - zC

        new_vec = [xD, yD, zD, xC, yC, zC]
        scaled_tip_to_tail_vectors.append(new_vec)
    return np.array(scaled_tip_to_tail_vectors)


def generate_bild_file(
    tip_to_tail_vectors,
    transformed_field_array,
    percentile,
    sparsify_factor,
    output,
    file_path,
):
    """
    This function generates the BILD file for the 3D vector field
    Inputs:
        tip_to_tail_vectors: Tip to tail vectors
        transformed_field_array: Field array
        chimera_type: Chimera type
        percentile: Percentile cutoff for sparsifying the field
        sparsify_factor: Sparsification factor for the field
    Outputs:
        None
    """

    with open(file_path, "r") as file:
        first_line = file.readline().strip()

    # Extract the part of the line that contains the Sample Density values
    density = first_line.split(";")[0].split(":")[1].strip()
    dim = first_line.split(";")[1].split(":")[2].strip()

    # Split the values and convert them to integers
    density1, density2, density3 = map(int, density.split())
    dim1, dim2, dim3 = map(float, dim.split())

    nx, ny, nz = density1, density2, density3

    # Added code to deal with non-ideal sparsification values: take s and convert it to hcf(closest(factors(nx),s),closest(factors(ny),s),closest(factors(ny),s))i
    sparsify_factor = math.gcd(
        fac(nx, sparsify_factor), fac(nx, sparsify_factor), fac(nx, sparsify_factor)
    )
    print(
        "Sparsification factor modified to: ",
        sparsify_factor,
        " to give grid-like visualization.",
    )

    transformed_field_vecs = transformed_field_array[:, 3:6]
    field_mags = np.linalg.norm(transformed_field_vecs, axis=1)
    field_mags_max = np.max(field_mags)
    ##field_mags_max = np.partition(field_mags.flatten(),-10)[-10]
    # field_mags_max = 0.10
    # field_mags_min = 0.001
    field_mags_min = np.min(field_mags)
    field_mags = (field_mags - field_mags_min) / field_mags_max
    percentile_cutoff = np.percentile(field_mags, percentile)
    r = sparsify_vec_field(
        (1 - field_mags), sparsify_factor, nx, ny, nz, dim1, dim2, dim3
    )
    # r = (np.ones((field_mags.shape[0], 1))).astype(int)[::sparsify_factor]
    g = sparsify_vec_field(
        (1 - field_mags), sparsify_factor, nx, ny, nz, dim1, dim2, dim3
    )
    # g = (1-field_mags)[::sparsify_factor]
    b = sparsify_vec_field(
        (np.ones((field_mags.shape[0], 1))).astype(int),
        sparsify_factor,
        nx,
        ny,
        nz,
        dim1,
        dim2,
        dim3,
    )
    field_mags = sparsify_vec_field(
        field_mags, sparsify_factor, nx, ny, nz, dim1, dim2, dim3
    )
    tip_to_tail_vectors = sparsify_vec_field(
        tip_to_tail_vectors, sparsify_factor, nx, ny, nz, dim1, dim2, dim3, 1
    )

    tip_to_tail_vectors = scale_tip_to_tail_vectors(
        tip_to_tail_vectors, field_mags, sparsify_factor, min(dim1, dim2, dim3)
    )

    with open(f"{output}.bild", "w") as bild:
        bild.write(".transparency 0.25\n")
        # for i in range(tip_to_tail_vectors.shape[0]):
        for i in range(len(tip_to_tail_vectors)):
            if field_mags[i] > percentile_cutoff:
                bild.write(f".color {r[i]} {g[i]} {b[i][0]}\n")
                bild.write(
                    f".arrow {tip_to_tail_vectors[i, 0]} {tip_to_tail_vectors[i, 1]} {tip_to_tail_vectors[i, 2]} {tip_to_tail_vectors[i, 3]} {tip_to_tail_vectors[i, 4]} {tip_to_tail_vectors[i, 5]} 0.01 {0.04*field_mags[i]*sparsify_factor*2.5*min(dim1,dim2,dim3)} 0.001\n"
                )
    bild.close()
