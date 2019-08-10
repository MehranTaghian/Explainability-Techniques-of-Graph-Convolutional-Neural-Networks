from torch.utils.data import Dataset
import dgl
import os
from CreateGraphEdge import get_edge_list
import torch

DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome"
NODES_DIR = DATA_DIR + '\\Features'
EDGES_DIR = DATA_DIR + '\\Edges'
REGIONS_DIR = DATA_DIR + '\\Regions'


class GraphDataset(Dataset):
    """
    experiment: can be 'countryVSurban' or 'indoorVSoutdoor'
    """

    def __init__(self, experiment):
        self.sample = []
        graph_list = os.listdir(NODES_DIR + F'\\{experiment}')
        self.ids = []
        for file in graph_list:
            features = torch.load(NODES_DIR + F'\\{experiment}\\{file}')
            image_id = file.split('-')[2]
            label = int((file.split('-')[3]).split('.')[0])
            edge = get_edge_list(image_id)
            g = dgl.DGLGraph()
            g.add_nodes(features.shape[0])
            g.add_edges(u=edge[:, 0], v=edge[:, 1])
            g.ndata['h'] = features
            self.ids.append(image_id)
            self.sample.append((g, label))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, item):
        return self.sample[item]
