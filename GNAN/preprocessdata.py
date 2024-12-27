import os
import torch
import dgl

def preProcessData(data, data_name="mutagenicity", processed_data_dir="processed_data"):
    # Unpack the data into graphs and labels
    graphs, labels = zip(*data)
    print(type(graphs))
    # Preprocess each graph in the dataset
    for graph in graphs:
        # Compute the shortest distance matrix
        shortest_dist_matrix = dgl.shortest_dist(graph)

        # Calculate normalization distance matrix (binary matrix where 1 means the nodes are connected)
        normalization_distance_matrix = 1 * (
            shortest_dist_matrix[..., :, None] == shortest_dist_matrix[..., None, :]
        ).sum(-1)

        # Calculate distance matrix (inverse of shortest distance)
        distance_matrix = 1 / (1 + shortest_dist_matrix)

        # Add these matrices as node features
        graph.ndata["normalization_distance_matrix"] = normalization_distance_matrix
        graph.ndata["distance_matrix"] = distance_matrix.float()

    # Ensure the directory exists for saving the processed data
    os.makedirs(processed_data_dir, exist_ok=True)

    # Define the path to save the entire dataset
    processed_data_path = os.path.join(processed_data_dir, f"{data_name}_processed")

    # Optionally save labels along with graphs (in a dictionary)
    labels_dict = {"labels": torch.tensor(labels)}

    # Save the entire dataset (graphs + labels) using dgl.save_graphs
    dgl.save_graphs(
        processed_data_path + "_graphs.bin", list(graphs), labels=labels_dict
    )

    print(f"Entire processed dataset saved to {processed_data_path}_graphs.bin")
