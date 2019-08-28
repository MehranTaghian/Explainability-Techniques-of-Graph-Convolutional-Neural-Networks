from torch.utils.data import Dataset
import dgl
import os
from CreateGraphEdge import get_edge_list
import torch
from Federico.torchgraphs.src import torchgraphs as tg

DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome"
NODES_DIR = DATA_DIR + '\\Features'


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
            # g = dgl.DGLGraph()
            # g.add_nodes(features.shape[0])
            # g.add_edges(u=edge[:, 0], v=edge[:, 1])
            # g.ndata['h'] = features
            g = tg.Graph(
                node_features=features,
                senders=torch.tensor(edge[:, 0]),
                receivers=torch.tensor(edge[:, 1])
            )

            self.ids.append(image_id)
            self.sample.append((g, label))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, item):
        return self.sample[item]
