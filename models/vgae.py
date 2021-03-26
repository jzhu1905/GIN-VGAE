# code modified based on https://github.com/zfjsail/gae-pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, device):
        super(GCNModelVAE, self).__init__()
        # self.gin = GraphCNN()
        self.device = device
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, batch_graph):
        recovered_adjs = []
        mus, logvars = [], []
        for graph in batch_graph:
            # adj =
            # node_features =
            mu, logvar = self.encode(graph.node_features, graph.adj)
            z = self.reparameterize(mu, logvar)
            recovered_adj = self.dc(z)
            recovered_adjs.append(recovered_adj)
            mus.append(mu)
            logvars.append(logvar)
        return recovered_adjs, mus, logvars

    # def forward(self, x, adj):
    #     mu, logvar = self.encode(x, adj)
    #     z = self.reparameterize(mu, logvar)
    #     return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
