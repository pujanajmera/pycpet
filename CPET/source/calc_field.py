from CPET.source.calculator import calculator


def main():
    options = {
        "path_to_pdb": "../../tests/test_files/1XHO-run1-328.pdb",
        "center": [40.183,  25.665,  23.780],
        "x": [41.183,  25.665,  23.780],
        "y": [40.183,  26.665,  23.780],
        "n_samples": 1000,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.1,
        "batch_size": 10,
        "concur_slip": 16,
        "filter_radius": 20.0,
        "filter_in_box": True, 
        "check_interval": 10
        #"filter_resids": ["HEM"]
    }

    calc_20 = calculator(options)
    print(calc_20.compute_box().shape)
    print(calc_20.compute_topo().shape)


main()