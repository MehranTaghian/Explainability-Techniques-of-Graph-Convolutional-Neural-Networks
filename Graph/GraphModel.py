import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(2048, 128, F.relu)
        self.gcn2 = GCN(128, 256, F.relu)
        self.gcn3 = GCN(256, 512, F.relu)
        self.gcn4 = GCN(2, 256, F.softmax)
        # self.gap = GAP()  # creates 2 classes for country vs urban or indoor vs outdoor

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        x = self.gcn3(g, x)
        x = self.gcn4(g, x)
        # x = self.gap(g, x)
        return x


# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
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
            GCN(256, 512, F.relu)
        ])
        self.classify = nn.Linear(512, n_classes)

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

# class NodeApplyModule(nn.Module):
#     def __init__(self, in_feats, out_feats, activation):
#         super(NodeApplyModule, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#         self.activation = activation
#
#     def forward(self, node):
#         h = self.linear(node.data['h'])
#         h = self.activation(h)
#         return {'h': h}
#
#
# class NodeGlobalAveragePooling(nn.Module):
#     def __init__(self, in_feats):
#         super(NodeGlobalAveragePooling, self).__init__()
#         self.gap = nn.AvgPool1d(kernel_size=in_feats, stride=1)  # Output features will be equal to 1
#         self.activation = F.softmax  # Activation is softmax
#
#     def forward(self, node):
#         h = self.gap(node.data['h'])
#         h = self.activation(h)
#         return {'h': h}
#
#
# class GCN(nn.Module):
#     def __init__(self, in_feats, out_feats, activation):
#         super(GCN, self).__init__()
#         self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
#
#     def forward(self, g, feature):
#         g.ndata['h'] = feature
#         g.update_all(gcn_msg, gcn_reduce)
#         g.apply_nodes(func=self.apply_mod)
#         return g.ndata.pop('h')
#
#
# class GAP(nn.Module):
#     def __init__(self, in_feats):
#         super(GAP, self).__init__()
#         self.apply_mod = NodeGlobalAveragePooling(in_feats)
#
#     def forward(self, g, feature):
#         g.ndata['h'] = feature
#         g.update_all(gcn_msg, gcn_reduce)
#         g.apply_nodes(func=self.apply_mod)
#         return g.ndata.pop('h')
