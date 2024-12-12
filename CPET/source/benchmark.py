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
