import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    # accum = torch.mean(nodes.mailbox['m'], 1)
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        self.X = h
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        self.X = g.ndata['h']
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Classifier(nn.Module):
    def __init__(self, in_dim=2048, n_classes=2):
        super(Classifier, self).__init__()
        self.Z = []
        self.layers = nn.ModuleList([
            GCN(in_dim, 128, F.relu),
            GCN(128, 256, F.relu),
            GCN(256, 512, F.relu),
            GCN(512, 256, F.relu),
            GCN(256, 128, F.relu)
        ])
        self.classify = nn.Linear(128, n_classes)

    def forward(self, g):
        self.Z = []
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)
            self.Z.append(h)
        g.ndata['h'] = h
        self.X = dgl.mean_nodes(g, 'h')
        self.Z.append(self.classify(self.X))
        return self.Z[-1]
