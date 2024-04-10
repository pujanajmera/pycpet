import numpy as np
from CPET.source.calculator import calculator
import warnings 
warnings.filterwarnings(action='ignore')
from scipy.stats import chisquare
from scipy.stats import entropy
import matplotlib.pyplot as plt 
import matplotlib
from copy import deepcopy


def mean_and_curve_to_hist(mean_dist, curve): 
    #Calculate reasonable maximum distances and curvatures
    #curvatures, distances = [],[]
    max_distance = max(mean_dist)
    max_curvature = max(curve)
    
    # bins is number of histograms bins in x and y direction (so below is 200x200 bins)
    # range gives xrange, yrange for the histogram
    a, b, c, q = plt.hist2d(
        mean_dist,
        curve,
        bins=100,
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


def main():
    options = {
        "path_to_pqr": "../../tests/test_files/test_large.pqr",
        "center": [104.785, 113.388, 117.966],
        "x": [105.785, 113.388, 117.966],
        "y": [104.785, 114.388, 117.966],
        "n_samples": 100000,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.01,
        "batch_size": 10,
        "concur_slip": 16,
        "filter_radius": 20.0,
        "filter_in_box": True, 
        "check_interval": 10
        #"filter_resids": ["HEM"]
    }

    topo_20 = calculator(options)
    ret = topo_20.compute_topo()
    dist_c = ret[0] 
    curve_c = ret[1]
    hist_20 = mean_and_curve_to_hist(dist_c, curve_c)
    

    options_30 = deepcopy(options)
    options_30["filter_radius"] = 30.0
    topo_30 = calculator(options_30)
    ret = topo_30.compute_topo()
    dist_c = ret[0] 
    curve_c = ret[1]
    hist_30 = mean_and_curve_to_hist(dist_c, curve_c)

    options_40 = deepcopy(options)
    options_40["filter_radius"] = 40.0
    topo_40 = calculator(options_40)
    ret = topo_40.compute_topo()
    dist_c = ret[0] 
    curve_c = ret[1]
    hist_40 = mean_and_curve_to_hist(dist_c, curve_c)

    options_50 = deepcopy(options)
    options_50["filter_radius"] = 50.0
    topo_50 = calculator(options_50)
    ret = topo_50.compute_topo()
    dist_c = ret[0] 
    curve_c = ret[1]
    hist_50 = mean_and_curve_to_hist(dist_c, curve_c)

    options_200 = deepcopy(options)
    options_200["filter_radius"] = 200.0
    topo_200 = calculator(options_200)
    ret = topo_200.compute_topo()
    dist_c = ret[0] 
    curve_c = ret[1]
    hist_200 = mean_and_curve_to_hist(dist_c, curve_c)

    distance_numpy_20_200 = distance_numpy(hist_20, hist_200)
    distance_numpy_30_200 = distance_numpy(hist_30, hist_200)
    distance_numpy_40_200 = distance_numpy(hist_40, hist_200)
    distance_numpy_50_200 = distance_numpy(hist_50, hist_200)

    distance_numpy_20_30 = distance_numpy(hist_20, hist_30)
    distance_numpy_30_40 = distance_numpy(hist_30, hist_40)
    distance_numpy_40_50 = distance_numpy(hist_40, hist_50)
    
    print("dist 20 --> 200: {}".format(distance_numpy_20_200))
    print("dist 30 --> 200: {}".format(distance_numpy_30_200))
    print("dist 40 --> 200: {}".format(distance_numpy_40_200))
    print("dist 50 --> 200: {}".format(distance_numpy_50_200))
    print("dist 20 --> 30: {}".format(distance_numpy_20_30))
    print("dist 30 --> 40: {}".format(distance_numpy_30_40))
    print("dist 40 --> 50: {}".format(distance_numpy_40_50))
    
main()