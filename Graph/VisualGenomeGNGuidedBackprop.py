from Federico.src import guidedbackprop as gp
from .VisualGenomeGraphModel import VisualGenomeGN
import torch.nn as nn
from collections import OrderedDict


class VisualGenomeGNGuidedBP(VisualGenomeGN):
    def __init__(self, in_dim=2048):
        super(VisualGenomeGN, self).__init__()
        self.layers = nn.Sequential(OrderedDict({
            'node1': gp.NodeLinearGuidedBP(128, node_features=in_dim, aggregation='avg'),
            'node1_relu': gp.NodeReLUGuidedBP(),
            'node2': gp.NodeLinearGuidedBP(256, node_features=128, aggregation='avg'),
            'node2_relu': gp.NodeReLUGuidedBP(),
            'node3': gp.NodeLinearGuidedBP(512, node_features=256, aggregation='avg'),
            'node3_relu': gp.NodeReLUGuidedBP(),
            'nodes_to_global': gp.GlobalLinearGuidedBP(2, node_features=512, aggregation='avg')
        }))
