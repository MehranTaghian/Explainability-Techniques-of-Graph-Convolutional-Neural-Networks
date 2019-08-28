from collections import OrderedDict
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
from Federico.torchgraphs.src import torchgraphs as tg


class VisualGenomeGN(nn.Module):
    def __init__(self, in_dim=2048, n_classes=2):
        super(VisualGenomeGN, self).__init__()
        self.Z = []
        self.layers = nn.Sequential(OrderedDict({
            'node1': tg.NodeLinear(128, node_features=in_dim, aggregation='avg'),
            'node1_relu': tg.NodeReLU(),
            'node2': tg.NodeLinear(256, node_features=128, aggregation='avg'),
            'node2_relu': tg.NodeReLU(),
            'node3': tg.NodeLinear(512, node_features=256, aggregation='avg'),
            'node3_relu': tg.NodeReLU(),
            'nodes_to_global': tg.GlobalLinear(2, node_features=512, aggregation='avg')
        }))
        # self.Global_avg_Pooling = tg.NodesToGlobal('avg')
        # self.Classifier = nn.Linear(512, n_classes)

    def forward(self, g):
        g = self.layers(g)
        return g
