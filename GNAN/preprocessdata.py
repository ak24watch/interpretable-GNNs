import dgl
import torch
import os


def preProcessData(data, data_name="mutagenicity", processed_data_dir="processed_data"):
    graphs, labels = zip(*data)
    for graph in graphs:
        shortest_dist_matrix = dgl.shortest_dist(graph)
        normalization_distance_matrix = 1 * (
            shortest_dist_matrix[..., :, None] == shortest_dist_matrix[..., None, :]
        ).sum(-1)
        distance_matrix = 1 / (1 + shortest_dist_matrix)

        graph.ndata["normalization_distance_matrix"] = normalization_distance_matrix
        graph.ndata["distance_matrix"] = distance_matrix.float()

    if not os.path.exists(f"{processed_data_dir}/{data_name}.pt"):
        os.makedirs(processed_data_dir)
        torch.save(data, f"{processed_data_dir}/{data_name}.pt")
        print(f"Saved preprocessed {data_name} dataset")

