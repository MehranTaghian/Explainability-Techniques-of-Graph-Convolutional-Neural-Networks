from collections import OrderedDict
from .VisualGenomeGraphModel import VisualGenomeGN
import torch
import Federico.src.torchgraphs as tg
import torch.nn as nn
import Federico.src.relevance as relevance


class VisualGenomeGNRelevance(VisualGenomeGN):
    def __init__(self, aggregation, bias, in_dim=2048):
        super(VisualGenomeGN, self).__init__()
        self.layers = nn.Sequential(OrderedDict({
            'node1': tg.NodeLinear(128, node_features=in_dim, aggregation='avg'),
            'node1_relu': tg.NodeReLU(),
            'node2': tg.NodeLinear(256, node_features=128, aggregation='avg'),
            'node2_relu': tg.NodeReLU(),
            'node3': tg.NodeLinear(512, node_features=256, aggregation='avg'),
            'node3_relu': tg.NodeReLU(),
            'nodes_to_global': tg.GlobalLinear(2, node_features=512, aggregation='avg')
        }))
