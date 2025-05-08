import numpy as np
from CPET.source.calculator import calculator
import warnings

warnings.filterwarnings(action="ignore")
from scipy.stats import chisquare
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib


def mean_and_curve_to_hist(mean_dist, curve):
    # Calculate reasonable maximum distances and curvatures
    # curvatures, distances = [],[]
    max_distance = max(mean_dist)
    max_curvature = max(curve)

    # bins is number of histograms bins in x and y direction (so below is 200x200 bins)
    # range gives xrange, yrange for the histogram
    a, b, c, q = plt.hist2d(
        mean_dist,
        curve,
        bins=50,
        range=[[0, max_distance], [0, max_curvature]],
        norm=matplotlib.colors.LogNorm(),
        density=True,
        cmap="jet",
    )

    NormConstant = 0
    for j in a:
        for m in j:
            NormConstant += m

    actual = []
    for j in a:
        actual.append([m / NormConstant for m in j])

    actual = np.array(actual)
    histogram = actual.flatten()
    return np.array(histogram)


def distance_numpy(hist1, hist2):
    a = (hist1 - hist2) ** 2
    b = hist1 + hist2
    return np.sum(np.divide(a, b, out=np.zeros_like(a), where=b != 0)) / 2.0


class Test_topos:
    def __init__(self):
        self.options = {
            "CPET_method": "woohoo",
            "path_to_pqr": "./test_files/test_large.pqr",
            "center": [104.785, 113.388, 117.966],
            "x": [105.785, 113.388, 117.966],
            "y": [104.785, 114.388, 117.966],
            "n_samples": 1000,
            "dimensions": [1.5, 1.5, 1.5],
            "step_size": 0.01,
            "batch_size": 10,
            "concur_slip": 16,
            "filter_radius": 50.0,
            "filter_in_box": True,
            "initializer": "uniform",
            # "filter_resids": ["HEM"]
        }
        self.topo = calculator(self.options, path_to_pdb=self.options["path_to_pqr"])

        ret = self.topo.compute_topo()
        self.dist_c = ret[0]
        self.curve_c = ret[1]

        ret2 = self.topo.compute_topo_batched()
        self.dist_batched = ret2[0]
        self.curve_batched = ret2[1]

        ret3 = self.topo.compute_topo_base()
        self.dist_base = ret3[0]
        self.curve_base = ret3[1]

    def test_topo_batch(self):
        hist = mean_and_curve_to_hist(self.dist_c, self.curve_c)
        hist2 = mean_and_curve_to_hist(self.dist_batched, self.curve_batched)
        print(distance_numpy(hist, hist2))

    def test_topo_cshared(self):
        hist = mean_and_curve_to_hist(self.dist_c, self.curve_c)
        hist2 = mean_and_curve_to_hist(self.dist_base, self.curve_base)
        print(distance_numpy(hist, hist2))

    def test_topo_batch_base(self):
        hist = mean_and_curve_to_hist(self.dist_batched, self.curve_batched)
        hist2 = mean_and_curve_to_hist(self.dist_base, self.curve_base)

        print(distance_numpy(hist, hist2))


test = Test_topos()
test.test_topo_batch()
test.test_topo_cshared()
test.test_topo_batch_base()
