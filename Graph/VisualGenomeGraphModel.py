from collections import OrderedDict
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
from TorchGraph.torchgraphs.src import torchgraphs as tg


class VisualGenomeGN(nn.Module):
    def __init__(self, in_dim=2048, n_classes=2):
        super(VisualGenomeGN, self).__init__()
        self.Z = []
        self.layers = nn.Sequential(OrderedDict({
            'edge1': tg.EdgeLinear(2048, sender_features=2048),
            'edge1_relu': tg.EdgeReLU(),
            'node1': tg.NodeLinear(128, node_features=in_dim, incoming_features=in_dim, aggregation='avg'),
            'node1_relu': tg.NodeReLU(),
            'edge2': tg.EdgeLinear(128, sender_features=128),
            'edge2_relu': tg.EdgeReLU(),
            'node2': tg.NodeLinear(256, node_features=128, incoming_features=128, aggregation='avg'),
            'node2_relu': tg.NodeReLU(),
            'edge3': tg.EdgeLinear(256, sender_features=256),
            'edge3_relu': tg.EdgeReLU(),
            'node3': tg.NodeLinear(512, node_features=256, incoming_features=256, aggregation='avg'),
            'node3_relu': tg.NodeReLU(),
            'nodes_to_global': tg.GlobalLinear(2, node_features=512, aggregation='avg')
        }))
        # self.Global_avg_Pooling = tg.NodesToGlobal('avg')
        # self.Classifier = nn.Linear(512, n_classes)

    def forward(self, g):
        # for l in self.layers:
        g = self.layers(g)
        return g
