# code modified based on https://github.com/weihua916/powerful-gnns
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN
from models.vgae import GCNModelVAE
from models.loss import loss_function


def train_gin(args, model, device, train_graphs, optimizer, epoch, loss_fn):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for _ in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = loss_fn(output, labels)
        print(output.shape, labels.shape)
        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


def train_vgae(args, model, device, train_graphs, optimizer, epoch, loss_fn):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for _ in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        recovered_adjs, mus, logvars = model(batch_graph)

        original_adjs = [train_graphs[idx].adj for idx in selected_idx]
        n_nodes = [train_graphs[idx].node_features.shape[0] for idx in selected_idx]
        norms = [adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
                 for adj in original_adjs]
        # recovered_adjs, mus, logvars = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = loss_fn(recovered_adjs, original_adjs, mus, logvars, n_nodes, norms, device)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


# pass data to model with mini-batch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(model, device, train_graphs, test_graphs):
    model.eval()

    res = []
    for graphs in [train_graphs, test_graphs]:
        output = pass_data_iteratively(model, graphs)
        pred = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        acc = correct / float(len(graphs))
        res.append(acc)

    print("accuracy train: %f test: %f" % (res[0], res[1]))

    return res


def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--vgae_hidden_dim', type=int, default=16,
                        help='number of hidden units for vage (default: 16)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--vage_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. '
                             'Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    args = parser.parse_args()

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)
    # 10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
    #
    # gin_model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim,
    #                      num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
    #                      args.neighbor_pooling_type, device).to(device)
    #
    # optimizer = optim.Adam(gin_model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # gin_loss = nn.CrossEntropyLoss()
    #
    # for epoch in range(1, args.epochs + 1):
    #     scheduler.step()
    #
    #     avg_loss = train_gin(args, gin_model, device, train_graphs, optimizer, epoch, gin_loss)
    #     acc_train, acc_test = test(gin_model, device, train_graphs, test_graphs)
    #
    #     if not args.filename == "":
    #         with open(args.filename, 'w') as f:
    #             f.write("Average Loss: %f Training Accuracy: %f Test Accuracy: %f" % (avg_loss, acc_train, acc_test))
    #             f.write("\n")
    #     print("\n")
    #     print(gin_model.eps)

    input_feat_dim = train_graphs[0].node_features.shape[1]
    vgae_model = GCNModelVAE(input_feat_dim, args.hidden_dim, args.vgae_hidden_dim, args.vage_dropout, device).to(device)
    vage_loss = loss_function

    optimizer = optim.Adam(vgae_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        avg_loss = train_vgae(args, vgae_model, device, train_graphs, optimizer, epoch, vage_loss)
        # acc_train, acc_test = test(vgae_model, device, train_graphs, test_graphs)

        if not args.filename == "":
            with open(args.filename, 'w') as f:
                f.write("Average Loss: %f" % (avg_loss))
                f.write("\n")
        # print(model.eps)


if __name__ == '__main__':
    main()
