import os
import dgl

import dgl.data
import torch
from preprocessdata import preProcessData


def getData(data_name="mutagenicity", processed_data_dir="processed_data"):
    if os.path.exists(f"{processed_data_dir}/{data_name}.pt"):
        data = torch.load(f"{processed_data_dir}/{data_name}.pt")
    else:
        data = dgl.data.LegacyTUDataset("Mutagenicity")
        preProcessData(data)
    num_class = data.num_classes
    num_feats = data[0][0].ndata["feat"].shape[1]
    train_data, valid_data, test_data = dgl.data.utils.split_dataset(data, shuffle=True)
    train_loader = dgl.dataloading.GraphDataLoader(train_data, batch_size=1)
    valid_loader = dgl.dataloading.GraphDataLoader(valid_data, batch_size=1)
    test_loader = dgl.dataloading.GraphDataLoader(test_data, batch_size=1)
    return train_loader, valid_loader, test_loader, num_feats, num_class
