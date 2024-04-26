import numpy as np
from CPET.source.calculator import calculator
from CPET.utils.calculator import distance_numpy, mean_and_curve_to_hist
import warnings 
warnings.filterwarnings(action='ignore')
from scipy.stats import chisquare
from scipy.stats import entropy
import matplotlib.pyplot as plt 
import matplotlib
from copy import deepcopy
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='CPET: A tool for computing and analyzing electric fields in proteins')
    parser.add_argument('-o', type=json.loads, help='Options for CPET', default=json.load(open("./options/options.json")))
    parser.add_argument('-pdb', type=str, help='Path to pdb file', default="./tests/1a0m.pdb")
    args = parser.parse_args()
    options = args.o
    file = args.pdb
    '''
    options_20 = deepcopy(options)
    options_20["filter_radius"] = 20.0
    topo_20 = calculator(options_20, path_to_pdb = file)
    ret = topo_20.compute_topo_GPU_batch_filter()
    dist_c = ret[:,0] 
    curve_c = ret[:,1]
    hist_20 = mean_and_curve_to_hist(dist_c, curve_c)
    np.savetxt("test20.top", ret)

    options_30 = deepcopy(options)
    options_30["filter_radius"] = 30.0
    topo_30 = calculator(options_30, path_to_pdb = file)
    ret = topo_30.compute_topo_GPU_batch_filter()
    dist_c = ret[:,0]  
    curve_c = ret[:,1]
    hist_30 = mean_and_curve_to_hist(dist_c, curve_c)
    
    options_40 = deepcopy(options)
    options_40["filter_radius"] = 40.0
    topo_40 = calculator(options_40, path_to_pdb = file)
    ret = topo_40.compute_topo_GPU_batch_filter()
    dist_c = ret[:,0]   
    curve_c = ret[:,1]
    hist_40 = mean_and_curve_to_hist(dist_c, curve_c)

    options_50 = deepcopy(options)
    options_50["filter_radius"] = 50.0
    topo_50 = calculator(options_50, path_to_pdb = file)
    ret = topo_50.compute_topo_GPU_batch_filter()
    dist_c = ret[:,0]   
    curve_c = ret[:,1]
    hist_50 = mean_and_curve_to_hist(dist_c, curve_c)
    '''
    options_200 = deepcopy(options)
    options_200["filter_radius"] = 200.0
    topo_200 = calculator(options_200, path_to_pdb = file)
    ret = topo_200.compute_topo_GPU_batch_filter()
    dist_c = ret[:,0]   
    curve_c = ret[:,1]
    hist_200 = mean_and_curve_to_hist(dist_c, curve_c)

    topo_200_1 = calculator(options_200, path_to_pdb = file)
    ret = topo_200_1.compute_topo_GPU_batch_filter()
    dist_c = ret[:,0]   
    curve_c = ret[:,1]
    hist_200_1 = mean_and_curve_to_hist(dist_c, curve_c)
    print(hist_200_1.shape)

    '''

    distance_numpy_20_200 = distance_numpy(hist_20, hist_200)
    distance_numpy_30_200 = distance_numpy(hist_30, hist_200)
    distance_numpy_40_200 = distance_numpy(hist_40, hist_200)
    distance_numpy_50_200 = distance_numpy(hist_50, hist_200)

    distance_numpy_20_30 = distance_numpy(hist_20, hist_30)
    distance_numpy_30_40 = distance_numpy(hist_30, hist_40)
    distance_numpy_40_50 = distance_numpy(hist_40, hist_50)
    distance_numpy_50_200 = distance_numpy(hist_50, hist_200)
    '''
    distance_numpy_200_200 = distance_numpy(hist_200, hist_200_1)
    
    '''
    print("dist 20 --> 200: {}".format(distance_numpy_20_200))
    print("dist 30 --> 200: {}".format(distance_numpy_30_200))
    print("dist 40 --> 200: {}".format(distance_numpy_40_200))
    print("dist 50 --> 200: {}".format(distance_numpy_50_200))
    print("dist 20 --> 30: {}".format(distance_numpy_20_30))
    print("dist 30 --> 40: {}".format(distance_numpy_30_40))
    print("dist 40 --> 50: {}".format(distance_numpy_40_50))
    '''

    print("dist 200 <-> 200: {}".format(distance_numpy_200_200))
    
main()