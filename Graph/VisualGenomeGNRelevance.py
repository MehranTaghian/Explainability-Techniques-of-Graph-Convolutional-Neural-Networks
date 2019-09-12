from collections import OrderedDict
from .VisualGenomeGraphModel import VisualGenomeGN

import torch.nn as nn
import Federico.src.relevance as relevance


class VisualGenomeGNRelevance(VisualGenomeGN):
    def __init__(self, aggregation, bias, in_dim=2048):
        super(VisualGenomeGN, self).__init__()
        self.layers = nn.Sequential(OrderedDict({
            'edge1': relevance.EdgeLinearRelevance(2048, sender_features=2048),
            'edge1_relu': relevance.EdgeReLURelevance(),
            'node1': relevance.NodeLinearRelevance(128, node_features=in_dim, incoming_features=in_dim, aggregation='avg'),
            'node1_relu': relevance.NodeReLURelevance(),
            'edge2': relevance.EdgeLinearRelevance(128, sender_features=128),
            'edge2_relu': relevance.EdgeReLURelevance(),
            'node2': relevance.NodeLinearRelevance(256, node_features=128, incoming_features=128, aggregation='avg'),
            'node2_relu': relevance.NodeReLURelevance(),
            'edge3': relevance.EdgeLinearRelevance(256, sender_features=256),
            'edge3_relu': relevance.EdgeReLURelevance(),
            'node3': relevance.NodeLinearRelevance(512, node_features=256, incoming_features=256, aggregation='avg'),
            'node3_relu': relevance.NodeReLURelevance(),
            'nodes_to_global': relevance.GlobalLinearRelevance(2, node_features=512, aggregation='avg')
        }))
