import os
import dgl
import random
import torch
from dgl.data import LegacyTUDataset
from preprocessdata import preProcessData


def getData(processed_data_dir="processed_data/mutagenicity_processed_graphs.bin"):
    if os.path.exists(processed_data_dir):
        graphs, label_dict = dgl.load_graphs(processed_data_dir)
    else:
        data = LegacyTUDataset("Mutagenicity")
        preProcessData(data)  # Your preprocessing function
        graphs, label_dict = dgl.load_graphs(processed_data_dir)

    # Get labels from the label_dict (assuming each graph corresponds to a label)
    labels = label_dict["labels"]

    # Randomly select 500 graphs from the dataset
    selected_indices = random.sample(range(len(graphs)), 500)
    selected_graphs = [graphs[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]

    # Split the selected 500 graphs into 300 for training, 100 for validation, and 100 for testing
    train_graphs, valid_graphs, test_graphs = (
        selected_graphs[:300],
        selected_graphs[300:400],
        selected_graphs[400:],
    )
    train_labels, valid_labels, test_labels = (
        selected_labels[:300],
        selected_labels[300:400],
        selected_labels[400:],
    )

    # Prepare the train, valid, and test data as pairs of graph and label
    train_data = list(zip(train_graphs, train_labels))
    valid_data = list(zip(valid_graphs, valid_labels))
    test_data = list(zip(test_graphs, test_labels))

    # Determine the number of features and classes
    num_feats = (
        selected_graphs[0].ndata["feat"].shape[1]
    )  # Assuming the first graph has the feature shape

    # Determine the number of classes based on the unique labels
    num_class = len(
        torch.unique(labels).tolist()
    )  # Use unique labels from the dataset

    return train_data, valid_data, test_data, num_feats, num_class
