import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def gen_param_dist_mat(dist_mat, topo_file_list):
    distances = pd.DataFrame(dist_mat)
    name = (
        topo_file_list[0].split("/")[-1].split("_")[0]
        + "_"
        + topo_file_list[0].split("/")[-1].split("_")[1]
        + "_"
        + topo_file_list[0].split("/")[-1].split("_")[2]
        + "_"
    )

    # Modify file names

    labels = topo_file_list
    labels = [
        label.replace(".top", "").split("/")[-1].replace(name, "") for label in labels
    ]

    # Map each label to its group
    group_map = {
        label: label.split("_")[-3] + "_" + label.split("_")[-2] for label in labels
    }
    grouped_labels = [group_map[label] for label in labels]
    print(group_map)
    print(grouped_labels)
    # Apply the new labels to the DataFrame
    distances.columns = grouped_labels
    distances.index = grouped_labels

    # Aggregate by taking the mean within each group for both rows and columns
    grouped = distances.groupby(level=0).mean()
    averaged_distances = grouped.T.groupby(level=0).mean()

    # Ensure the matrix is symmetric
    averaged_distances = (averaged_distances + averaged_distances.T) / 2

    # (Optional) Plot the distance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(averaged_distances, cmap="Greens_r", annot=True, linewidths=0.1)
    plt.title("Averaged Distance Matrix")
    plt.show()

    return averaged_distances


def analyze_param_dist_mat(averaged_distances, threshold=0.1, mode="optimal_field"):
    """
    Takes in averaged distance matrix and returns optimal list of parameters

    Inputs:
    distances: Averaged distance matrix in pandas format
    threshold: Threshold for the distance matrix to determine either self-consistency
    or converged field
    mode: Mode of analysis; either "self_consistency" or "optimal_field"

    Outputs:
    None (prints optimal parameters)
    """

    # Generate list of step sizes and samples sweeped over
    sample_list = []
    stepsize_list = []
    for i in reversed(list(averaged_distances.columns)):
        samples = int(i.split("_")[0])
        stepsize = float("0." + i.split("_")[1])
        if samples not in sample_list:
            sample_list.append(int(samples))
        if stepsize not in stepsize_list:
            stepsize_list.append(stepsize)

    i = 0
    updated_dist = 1
    while updated_dist > threshold:
        current_samples = sample_list[i]
        if mode == "optimal_field":
            current_stepsize = stepsize_list[0]
            if i < len(sample_list) - 1:
                # First determine if just increasing the number of samples while keeping stepsize at initial value achieves threshold
                updated_dist = averaged_distances.loc[
                    str(sample_list[i]) + "_" + str(stepsize_list[0])[2:],
                    str(sample_list[i + 1]) + "_" + str(stepsize_list[0])[2:],
                ]
                if updated_dist < threshold:
                    print(
                        f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}"
                    )
                    return
            # Now, go through list of stepsizes and see off-diagonal distances for threshold
            for j in range(len(stepsize_list) - 1):
                updated_dist = averaged_distances.loc[
                    str(sample_list[i]) + "_" + str(stepsize_list[j])[2:],
                    str(sample_list[i]) + "_" + str(stepsize_list[j + 1])[2:],
                ]
                if updated_dist < threshold:
                    current_samples = sample_list[i]
                    current_stepsize = stepsize_list[j]
                    print(
                        f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}"
                    )
                    return
                if i == len(sample_list) - 1 and j == len(stepsize_list) - 2:
                    current_samples = sample_list[i]
                    current_stepsize = stepsize_list[j + 1]
                    print(
                        f"Hit sample and stepsize limit, reporting highest sampling values. Next highest sampling values are convergent at distance {updated_dist}"
                    )
                    print(
                        f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}"
                    )
                    return
        if mode == "self_consistency":
            # Now, go through list of stepsizes and see diagonal distances for threshold
            for j in range(len(stepsize_list)):
                # Compare with the same i and updated j
                updated_dist = averaged_distances.loc[
                    str(sample_list[i]) + "_" + str(stepsize_list[j])[2:],
                    str(sample_list[i]) + "_" + str(stepsize_list[j])[2:],
                ]
                if updated_dist < threshold:
                    current_stepsize = stepsize_list[j]
                    print(
                        f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}"
                    )
                    return
                if i == len(sample_list) - 1 and j == len(stepsize_list) - 1:
                    current_stepsize = stepsize_list[j]
                    print(
                        f"Hit sample and stepsize limit, reporting highest sampling values. Final distance is at {updated_dist}"
                    )
                    print(
                        f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}"
                    )
                    return
        i += 1
