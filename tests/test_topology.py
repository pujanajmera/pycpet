import numpy as np
from CPET.source.calculator import calculator
import warnings 
warnings.filterwarnings(action='ignore')
from scipy.stats import chisquare
from scipy.stats import entropy
import matplotlib.pyplot as plt 
import matplotlib
#from pyinstrument import Profiler

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


def gather_reference_simplest_implementation_topo(topo):
    hist = topo.compute_topo_single()
    return hist

class Test_topos:
    options = {
        "center": {
                "method": "first",
                "atoms": {
                        "CD": 2
                }
        },
        "x": {
                "method": "mean",
                "atoms": {
                        "CG": 1,
                        "CB": 1
                }
        },
        "y": {
                "method": "inverse",
                "atoms": {
                        "CA": 3,
                        "CB": 3
                }
        },
        "n_samples": 1000,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.01,
        "batch_size": 10,
        "concur_slip": 12,
        "filter_radius": 40.0,
        "filter_in_box": True, 
        "initializer": "uniform",
        "CPET_method": "topology",
        "max_streamline_init": "fixed_rand"
        #"filter_resids": ["HEM"]
    }
    topo = calculator(options, path_to_pdb="./test_files/test_large.pdb")
    # change max_array to int 
    reference_hist = gather_reference_simplest_implementation_topo(topo)


    def topo_equality(self, test_topos):
        np.testing.assert_allclose(
            self.hist_base, 
            test_topos, rtol=1e-2, atol=1e-2)
        

    def test_topo_methods(self):
        topo_function_list = [
            self.topo.compute_topo,
            self.topo.compute_topo_complete_c_shared
        
        ]
        
        for topo_function in topo_function_list:
            print("----"*15)
            hist = topo_function()
            self.topo_equality(hist)


        # TODO: pujan add the topo for gpu - will need to batch the start points

#test = Test_topos()
#test.test_topo_methods()
#test.test_topo_batch()
#test.test_topo_cshared()
#test.test_topo_batch_base()


