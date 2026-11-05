import numpy as np
import warnings
import math
import logging

from CPET.utils.io import default_options_initializer

"""
Overall script for all visualization functions and visualization-affiliates.

Current functions:
- visualize_field / visualize_esp: Visualize electric fields or electrostatic
potentials in proteins by creating a bild file for Chimera/ChimeraX input

To-add:
- Time-dependent animations
- Clustering visualizations
- High quality distance matrix plotting (?)
"""

log = logging.getLogger(__name__)


def visualize_esp(path_to_esp, outputpath, options):
    """
    Write a BILD file showing an electrostatic potential as colored spheres, for Chimera/ChimeraX.

    Positive potentials are drawn from white to red and negative potentials from
    white to blue, with sphere size growing with magnitude.

    Parameters
    ----------
    path_to_esp : str
        Path to the electrostatic potential file.
    outputpath : str
        Folder to write ``<name>.bild`` to.
    options : dict
        Options; reads ``visualization["cutoff"]`` and the ``max_p_esp``,
        ``min_p_esp``, ``max_n_esp`` and ``min_n_esp`` color limits.

    Returns
    -------
    str
        A message naming the file that was saved.
    """
    # Fill defaults in case this is called outside of CPET.py
    options = default_options_initializer(options)

    cutoff = options["visualization"]["cutoff"]
    sparsify_factor = options["visualization"]["sparsify_factor"]

    max_p_esp = options["max_p_esp"]
    min_p_esp = options["min_p_esp"]
    max_n_esp = options["max_n_esp"]
    min_n_esp = options["min_n_esp"]

    name = path_to_esp.split("/")[-1].split(".")[0]

    log.info(f"Generating BILD file for {name}")

    bild_path = outputpath + "/" + name

    # Generate bild file
    (
        sample_density_array,
        volume_box_array,
        center_array,
        basis_matrix_array,
        esp_array,
    ) = process_field_file(path_to_esp, type="esp")

    esp_array = transform_esp(esp_array, center_array, basis_matrix_array)

    esp_radii = generate_radii(esp_array.copy(), sample_density_array, volume_box_array)

    # ESP is a scalar field, so no need to transform or generate vectors, represent as spheres
    generate_bild_file_esp(
        esp_radii,
        esp_array,
        cutoff,
        max_p_esp,
        min_p_esp,
        max_n_esp,
        min_n_esp,
        bild_path,
    )
    return f"Bild file saved for {name}"


def visualize_field(path_to_efield, outputpath, options):
    """
    Write a BILD file showing an electric field as arrows, for Chimera/ChimeraX.

    Parameters
    ----------
    path_to_efield : str
        Path to the electric field file.
    outputpath : str
        Folder to write ``<name>.bild`` to.
    options : dict
        Options; reads ``visualization["cutoff"]`` and
        ``visualization["sparsify_factor"]``.

    Returns
    -------
    str
        A message naming the file that was saved.
    """

    # Extract key options, filling defaults in case this is called outside of CPET.py
    options = default_options_initializer(options)

    cutoff = options["visualization"]["cutoff"]
    sparsify_factor = options["visualization"]["sparsify_factor"]

    name = path_to_efield.split("/")[-1].split(".")[0]

    log.info(f"Generating BILD file for {name}")

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
        tip_to_tail_vectors,
        transformed_field_array,
        cutoff,
        sparsify_factor,
        bild_path,
        path_to_efield,
    )
    return f"Bild file saved for {name}"


def process_field_file(file_path, type="field"):
    """
    Read the header and data of a field or electrostatic potential file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    type : str, optional
        ``"field"`` for electric field files (``x y z Ex Ey Ez`` rows) or
        ``"esp"`` for electrostatic potential files (``x y z V`` rows). Default
        is ``"field"``.

    Returns
    -------
    sample_density : np.ndarray
        Number of grid points along each axis, shape (3,).
    volume_box : np.ndarray
        Box half widths in Angstroms, shape (3,).
    center : np.ndarray
        Box center in Angstroms, shape (3,).
    basis_matrix : np.ndarray
        Basis matrix of the box frame, shape (3, 3).
    data : np.ndarray
        The data rows, shape (N, 6) for fields or (N, 4) for potentials.

    Raises
    ------
    ValueError
        If a header or data line is malformed, there is no data, or the number
        of rows doesn't match the sample density.
    """
    # Initialize variables
    sample_density = []
    volume_box = []
    center = []
    basis_matrix = []
    field = []
    esp = []

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
                    if type == "field":
                        field.append([float(x) for x in line.split()])
                    elif type == "esp":
                        esp.append([float(x) for x in line.split()])
                except:
                    raise ValueError(
                        f"Field/esp line not formatted correctly; field/esp list of length {len(field)}"
                    )
    if not sample_density:
        warnings.warn("Sample density not found, ignoring for checking")
    if not field and not esp:
        raise ValueError("No field/esp data found at all, exiting...")
    if not center or not basis_matrix:
        warnings.warn(
            "Center or basis matrix not found, no transformation will be made"
        )
    if sample_density:
        if type == "field":
            check_field(np.array(field), np.array(sample_density), type)
        elif type == "esp":
            check_field(np.array(esp), np.array(sample_density), type)
    log.debug("Basis matrix: %s", np.array(basis_matrix))
    if type == "field":
        return (
            np.array(sample_density),
            np.array(volume_box),
            np.array(center),
            np.array(basis_matrix),
            np.array(field),
        )
    elif type == "esp":
        return (
            np.array(sample_density),
            np.array(volume_box),
            np.array(center),
            np.array(basis_matrix),
            np.array(esp),
        )


def check_field(elec_array, sample_density_array, type="field"):
    """
    Check that the number of data rows matches the sample density.

    Parameters
    ----------
    elec_array : np.ndarray
        Field or potential rows, shape (N, 6) or (N, 4).
    sample_density_array : np.ndarray
        Number of grid points along each axis, shape (3,).
    type : str, optional
        ``"field"`` or ``"esp"``, used in the error message. Default is ``"field"``.

    Raises
    ------
    ValueError
        If N is not the product of the sample density.
    """
    log.debug("Sample density: %s", sample_density_array)
    log.debug("Shape of %s array: %s", type, elec_array.shape)
    if elec_array.shape[0] != np.product(sample_density_array, axis=0):
        raise ValueError(
            f"{type} provided does not match sample density, {type} of shape {elec_array.shape[0]} does not match expected sample amount of {np.product(sample_density_array, axis=0)}"
        )
    else:
        log.debug(f"Sample density check passed for {type} array")


def transform_field(field_array, center_array, basis_matrix_array):
    """
    Move field points from the box frame into the structure's frame.

    Coordinates are rotated by the basis matrix and shifted by the center;
    field vectors are only rotated.

    Parameters
    ----------
    field_array : np.ndarray
        Field rows ``x y z Ex Ey Ez`` in the box frame, shape (N, 6).
    center_array : np.ndarray
        Box center, shape (3,).
    basis_matrix_array : np.ndarray
        Basis matrix of the box frame, shape (3, 3).

    Returns
    -------
    transformed_field_array : np.ndarray
        Field rows in the structure's frame, shape (N, 6).
    """
    field_coords = field_array[:, 0:3]
    field_vecs = field_array[:, 3:6]
    log.debug(
        "Field coordinates: %s, field vectors: %s, basis matrix: %s",
        field_coords,
        field_vecs,
        basis_matrix_array,
    )
    # Need to transform and recenter field coords, but only transform field vecs
    transformed_field_coords = field_coords @ basis_matrix_array.T
    transformed_field_coords = transformed_field_coords + center_array
    transformed_field_vecs = np.matmul(field_vecs, basis_matrix_array.T)
    transformed_field_array = np.concatenate(
        (transformed_field_coords, transformed_field_vecs), axis=1
    )
    return transformed_field_array


def transform_esp(esp_array, center_array, basis_matrix_array):
    """
    Move potential points from the box frame into the structure's frame.

    Parameters
    ----------
    esp_array : np.ndarray
        Potential rows ``x y z V`` in the box frame, shape (N, 4).
    center_array : np.ndarray
        Box center, shape (3,).
    basis_matrix_array : np.ndarray
        Basis matrix of the box frame, shape (3, 3).

    Returns
    -------
    transformed_esp_array : np.ndarray
        Potential rows in the structure's frame, shape (N, 4).
    """
    esp_coords = esp_array[:, 0:3]
    # Need to transform and recenter field coords, but only transform field vecs
    transformed_esp_coords = esp_coords @ basis_matrix_array.T
    transformed_esp_coords = transformed_esp_coords + center_array
    transformed_esp_array = np.concatenate(
        (transformed_esp_coords, np.array(esp_array[:, 3]).reshape(-1, 1)), axis=1
    )
    return transformed_esp_array


def generate_tip_tail_vectors(field_array, sample_density_array, volume_box_array):
    """
    Turn field vectors into arrow start and end points of equal length.

    Each arrow is centered on its grid point, points along the field, and is
    0.9 times the smallest grid spacing long. The vector columns of
    ``field_array`` are overwritten.

    Parameters
    ----------
    field_array : np.ndarray
        Field rows ``x y z Ex Ey Ez``, shape (N, 6).
    sample_density_array : np.ndarray
        Number of grid points along each axis, shape (3,).
    volume_box_array : np.ndarray
        Box half widths, shape (3,).

    Returns
    -------
    tip_to_tail_vectors : np.ndarray
        Tail and tip coordinates of each arrow, shape (N, 6).
    """

    max_vector_length = 0.9 * np.min(2 * volume_box_array / sample_density_array)

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


def generate_radii(esp_array, sample_density_array, volume_box_array):
    """
    Compute a sphere radius for each potential point.

    Radii grow with the magnitude of the potential, up to half the smallest
    grid spacing.

    Parameters
    ----------
    esp_array : np.ndarray
        Potential rows ``x y z V``, shape (N, 4).
    sample_density_array : np.ndarray
        Number of grid points along each axis, shape (3,).
    volume_box_array : np.ndarray
        Box half widths, shape (3,).

    Returns
    -------
    radii : np.ndarray
        Sphere radius for each point, shape (N,).
    """
    log.debug(
        "Sample density: %s, volume box: %s", sample_density_array, volume_box_array
    )

    # Normalize esp_array (only esp values)
    zero_min_esps = np.abs(esp_array[:, 3]) - np.min(esp_array[:, 3])
    esps_norm = zero_min_esps / np.max(zero_min_esps)

    # Generate radii based on grid and esp values
    radii = (
        0.5 * np.min(2 * volume_box_array / sample_density_array) * np.abs(esps_norm)
    )

    return radii


def fac(num, f):
    """
    Find the factor of an integer closest to a target value.

    Parameters
    ----------
    num : int
        The integer to factor.
    f : float
        The target value.

    Returns
    -------
    int
        The factor of ``num`` closest to ``f``; the larger one on a tie.
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
    Keep a regular subset of the grid points, to draw fewer arrows.

    The step along each axis is the factor of that axis's point count closest
    to ``sparsify_factor``, adjusted for the box's aspect ratio.

    Parameters
    ----------
    vectors : array_like
        One value or row per grid point, in grid order, length nx * ny * nz.
    sparsify_factor : int
        Approximate step between kept points.
    nx, ny, nz : int
        Number of grid points along x, y and z.
    dim1, dim2, dim3 : float
        Box half widths along x, y and z.
    printflag : int, optional
        If 1, log the grid size before and after. Default is 0.

    Returns
    -------
    np.ndarray
        The values or rows at the kept grid points.
    """
    # sparsify_factor=3
    log.debug("Grid points (nx, ny, nz): %s, %s, %s", nx, ny, nz)

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

    log.debug("Sparsification steps (sx, sy, sz): %s, %s, %s", sx, sy, sz)
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
    log.debug("Indices kept after y/z sparsification: %s", indices_yzs)
    log.debug("Grid points in y after sparsification: %s", new_ny)
    indices_xyzs = []
    for i in range(len(indices_yzs)):
        if (i // (new_ny * new_nz)) % sx == 0:  # if (i // 7*7) % 3 == 0
            indices_xyzs.append(indices_yzs[i])
    new_nx = nx / sx

    if printflag == 1:
        log.info(f"Original field has {nx} x {ny} x {nz} points")
        log.info(
            f"Sparsified field has {int(new_nx)} x {int(new_ny)} x {int(new_nz)} points before filtering"
        )

    vectors_s = []
    for i in indices_xyzs:
        vectors_s.append(vectors[i])
    vectors_s_np = np.array(vectors_s)
    log.debug("Number of sparsified vectors: %s", len(vectors_s_np))
    return vectors_s_np


def scale_tip_to_tail_vectors(
    tip_to_tail_vectors, field_mags, sparsify_factor, min_dim
):
    """
    Scale each arrow about its midpoint by its field magnitude.

    Parameters
    ----------
    tip_to_tail_vectors : np.ndarray
        Tail and tip coordinates of each arrow, shape (N, 6).
    field_mags : np.ndarray
        Normalized field magnitudes, shape (N,).
    sparsify_factor : int
        Overall scale factor, so sparser grids get longer arrows.
    min_dim : float
        Smallest box half width, used for scaling.

    Returns
    -------
    np.ndarray
        Scaled tail and tip coordinates, shape (N, 6).
    """
    scaled_tip_to_tail_vectors = []
    for i in range(len(tip_to_tail_vectors)):

        # use field mag as scaling factor
        m = sparsify_factor * field_mags[i] * min_dim

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
    Write the BILD file of arrows for an electric field.

    Arrows are colored from white (weak) to blue (strong) and only drawn above
    the percentile cutoff of normalized field magnitude.

    Parameters
    ----------
    tip_to_tail_vectors : np.ndarray
        Tail and tip coordinates of each arrow, shape (N, 6).
    transformed_field_array : np.ndarray
        Field rows ``x y z Ex Ey Ez``, shape (N, 6).
    percentile : float
        Percentile (0-100) of field magnitude below which arrows are not drawn.
    sparsify_factor : int
        Approximate step between drawn grid points.
    output : str
        Output path without the ``.bild`` extension.
    file_path : str
        Path to the field file, whose header gives the grid size.
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

    # Added code to deal with non-ideal sparsification values: take s and convert it to hcf(closest(factors(nx),s),closest(factors(ny),s),closest(factors(nz),s))
    sparsify_factor = math.gcd(
        fac(nx, sparsify_factor), fac(ny, sparsify_factor), fac(nz, sparsify_factor)
    )
    log.info(
        f"Sparsification factor set to {sparsify_factor} for a grid-like visualization"
    )

    transformed_field_vecs = transformed_field_array[:, 3:6]
    field_mags = np.linalg.norm(transformed_field_vecs, axis=1)
    field_mags_max = np.max(field_mags)
    ##field_mags_max = np.partition(field_mags.flatten(),-10)[-10]
    # field_mags_max = 0.10
    # field_mags_min = 0.001
    field_mags_min = np.min(field_mags)

    log.info(f"Max field magnitude: {field_mags_max}")
    log.info(f"Min field magnitude: {field_mags_min}")

    field_mags = (field_mags - field_mags_min) / (field_mags_max - field_mags_min)
    # Make sure min and max are 0 and 1
    log.debug(
        "Normalized field magnitude max and min: %s, %s", max(field_mags), min(field_mags)
    )
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

        for i in range(len(tip_to_tail_vectors)):
            if field_mags[i] > percentile_cutoff:
                bild.write(f".color {r[i]} {g[i]} {b[i][0]}\n")
                bild.write(
                    f".arrow {tip_to_tail_vectors[i, 0]} {tip_to_tail_vectors[i, 1]} {tip_to_tail_vectors[i, 2]} {tip_to_tail_vectors[i, 3]} {tip_to_tail_vectors[i, 4]} {tip_to_tail_vectors[i, 5]} 0.01 {0.04*field_mags[i]*sparsify_factor*2.5*min(dim1,dim2,dim3)} 0.001\n"
                )
    bild.close()


def generate_bild_file_esp(
    radii, esp_array, percentile, max_p_esp, min_p_esp, max_n_esp, min_n_esp, output
):
    """
    Write the BILD file of spheres for an electrostatic potential.

    Positive potentials are colored from white to red and negative potentials
    from white to blue. Spheres are only drawn above the percentile cutoff of
    absolute potential.

    Parameters
    ----------
    radii : np.ndarray
        Sphere radius for each point, shape (N,).
    esp_array : np.ndarray
        Potential rows ``x y z V``, shape (N, 4).
    percentile : float
        Percentile (0-100) of absolute potential below which spheres are not drawn.
    max_p_esp, min_p_esp : float or None
        Color limits for positive potentials. None uses the data's range.
    max_n_esp, min_n_esp : float or None
        Color limits for negative potentials. None uses the data's range.
    output : str
        Output path without the ``.bild`` extension.
    """

    esp_array_mags = np.abs(esp_array[:, 3])
    percentile_cutoff = np.percentile(esp_array_mags, percentile)

    """
    Make red positive and blue negative.
    For positive esp values, values range from (0,0,0) to (1,0,0)
    For negative esp values, values range from (0,0,0) to (0,0,1)
    (0,0,0) is for esp values of 0
    (1,0,0) is for the most positive esp value
    (0,0,1) is for the most negative esp value
    """

    # Max p esp is highest magnitude positive esp, min p esp is lowest magnitude positive esp
    # Max n esp is highest magnitude negative esp, min n esp is lowest magnitude negative esp

    max_p_esp = max_p_esp
    min_p_esp = min_p_esp
    max_n_esp = max_n_esp
    min_n_esp = min_n_esp
    esp_temp_positive = esp_array[esp_array[:, 3] > 0]
    esp_temp_negative = esp_array[esp_array[:, 3] < 0]
    if max_p_esp == None:
        # Make max_p_esp None if esp_temp_positive is empty
        max_p_esp = (
            np.max(esp_temp_positive[:, 3]) if len(esp_temp_positive) > 0 else None
        )
    if min_p_esp == None:
        min_p_esp = (
            np.min(esp_temp_positive[:, 3]) if len(esp_temp_positive) > 0 else None
        )
    if max_n_esp == None:
        max_n_esp = (
            np.max(esp_temp_negative[:, 3]) if len(esp_temp_negative) > 0 else None
        )
    if min_n_esp == None:
        min_n_esp = (
            np.min(esp_temp_negative[:, 3]) if len(esp_temp_negative) > 0 else None
        )

    log.info(f"Positive ESP max: {max_p_esp}, min: {min_p_esp}")
    log.info(f"Negative ESP max: {max_n_esp}, min: {min_n_esp}")

    r = np.ones((len(esp_array), 1))
    g = np.ones((len(esp_array), 1))
    b = np.ones((len(esp_array), 1))

    for i in range(len(esp_array)):
        if esp_array[i, 3] > 0:
            g[i] = 1 - (esp_array[i, 3] - min_p_esp) / (max_p_esp - min_p_esp)
            b[i] = 1 - (esp_array[i, 3] - min_p_esp) / (max_p_esp - min_p_esp)
        elif esp_array[i, 3] < 0:
            g[i] = 1 - (esp_array[i, 3] - max_n_esp) / (min_n_esp - max_n_esp)
            r[i] = 1 - (esp_array[i, 3] - max_n_esp) / (min_n_esp - max_n_esp)
        else:
            r[i] = 1
            g[i] = 1
            b[i] = 1

    with open(f"{output}.bild", "w") as bild:
        bild.write(".transparency 0.25\n")
        for i in range(len(esp_array)):
            if esp_array_mags[i] > percentile_cutoff:
                bild.write(f".color {r[i][0]} {g[i][0]} {b[i][0]}\n")
                bild.write(
                    f".sphere {esp_array[i, 0]} {esp_array[i, 1]} {esp_array[i, 2]} {radii[i]}\n"
                )
