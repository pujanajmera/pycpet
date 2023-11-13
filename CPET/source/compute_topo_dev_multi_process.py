import numpy as np
from CPET.source.topo_calc import Topo_calc



def main():
    options = {
        "path_to_pqr": "../../tests/test_files/test_large.pqr",
        "center": [55.965, 46.219, 22.123],
        "x": [56.191, 48.344, 22.221],
        "y": [57.118, 46.793, 20.46],
        "n_samples": 10000,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.01,
        "concur_slip": 16,
    }

    topo = Topo_calc(options)
    hist = topo.compute_topo()
    np.savetxt("hist_cpet.txt", hist)


main()
