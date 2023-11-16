import numpy as np
from CPET.source.topo_calc import Topo_calc



def main():
    options = {
        "path_to_pqr": "../../tests/test_files/test_large.pqr",
        "center": [104.785, 113.388, 117.966],
        "x": [105.785, 113.388, 117.966],
        "y": [104.785, 114.388, 117.966],
        "n_samples": 100,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.01,
        "concur_slip": 16,
        "filter_radius": 1.0,
        "filter_resids": ["HEM"]
    }


    topo = Topo_calc(options)
    hist = topo.compute_topo()
    np.savetxt("hist_cpet.txt", hist)


main()
