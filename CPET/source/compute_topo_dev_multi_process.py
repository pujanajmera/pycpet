import numpy as np
from CPET.source.topo_calc import Topo_calc
from CPET.utils.parser import options_parsing
import argparse

def main():
    parser = argparse.ArgumentParser(description="Calculate topologies - multiprocessed")
    parser.add_argument('-o', '--options', type=str, help='Path to options file, otherwise reads ./options.json', default='./options.json')
    args = parser.parse_args()

    options = options_parsing(args.options)

    topo = Topo_calc(options)
    hist = topo.compute_topo()
    np.savetxt("hist_cpet.txt", hist)


main()
