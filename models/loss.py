# code modified based on https://github.com/zfjsail/gae-pytorch/

import torch
import torch.nn.modules.loss
import torch.nn as nn
import torch.nn.functional as F


def vgae_loss_function(recovered_adjs, adjs, mus, logvars, n_nodes, norms, device):
    losses = 0
    for idx in range(len(adjs)):
        recovered_adj, adj = recovered_adjs[idx], adjs[idx]
        mu, logvar = mus[idx], logvars[idx]
        norm = norms[idx]
        cost = norm * F.binary_cross_entropy_with_logits(recovered_adj, adj)
        n_node = n_nodes[idx]
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_node * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        losses += cost+KLD
    return losses / len(adjs)
