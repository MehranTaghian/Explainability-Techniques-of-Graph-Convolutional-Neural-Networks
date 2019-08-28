from collections import OrderedDict

import dgl
import torch
from GraphDataset import GraphDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Graph.VisualGenomeGraphModel import VisualGenomeGN
import torch.nn as nn
from Federico.torchgraphs.src import torchgraphs as tg

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(train_percent=80, experiment='countryVSurban'):
    samples = GraphDataset(experiment)
    return train_test_split(samples, train_size=train_percent, shuffle=True)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, label = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    # batched_graph = tg.GraphBatch(graphs, label)
    return batched_graph, torch.tensor(label).to(DEVICE)


def train_model(epoches=100, experiment='countryVSurban', model=None):
    # Create training and test sets.
    testset, trainset = load_data(experiment=experiment)
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    # data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
    #                          collate_fn=collate)
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                             collate_fn=tg.GraphBatch.collate)

    # g, _, _ = next(iter(data_loader))
    # print(type(g))
    # Create model
    # if model is None:
    #     model = Classifier()

    model = VisualGenomeGN()

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    model.to(DEVICE)
    model.train()

    # for iter, (bg, labels) in enumerate(data_loader):
    #     print(bg)
    #     print(labels)

    epoch_losses = []
    for epoch in range(epoches):
        epoch_loss = 0
        for iter, (bg, labels) in enumerate(data_loader):
            # print(torch.sum(bg.num_nodes_by_graph))
            # print(bg.node_index_by_graph)
            out = model(bg)
            labels = labels.to(DEVICE)
            loss = loss_func(out.global_features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
    return model, testset, trainset


def model_eval(model, test_set, train_set):
    model.eval()
    # Convert a list of tuples to two lists
    train_X, train_Y = map(list, zip(*train_set))
    test_X, test_Y = map(list, zip(*test_set))

    data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                             collate_fn=tg.GraphBatch.collate)

    correct = 0
    for iter, (bg, labels) in enumerate(data_loader):
        labels = labels.to(DEVICE)
        probs_train_Y = torch.softmax(model(bg).global_features, 1)
        # argmax_train_Y = torch.max(probs_train_Y, 1)[1].view(-1, 1)
        argmax_train_Y = torch.argmax(probs_train_Y, 1)
        temp = (labels == argmax_train_Y).sum().item()
        # print(argmax_train_Y.shape)
        # print(temp)
        correct += temp

    print('Accuracy of argmax predictions on the train set: {:4f}%'.format(
        (correct / len(train_Y) * 100)))

    data_loader = DataLoader(testset, batch_size=32, shuffle=True,
                             collate_fn=tg.GraphBatch.collate)

    correct = 0
    for iter, (bg, labels) in enumerate(data_loader):
        labels = labels.to(DEVICE)
        probs_test_Y = torch.softmax(model(bg).global_features, 1)
        # argmax_train_Y = torch.max(probs_train_Y, 1)[1].view(-1, 1)
        argmax_test_Y = torch.argmax(probs_test_Y, 1)
        temp = (labels == argmax_test_Y).sum().item()
        # print(argmax_train_Y.shape)
        # print(temp)
        correct += temp

    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (correct / len(test_Y) * 100)))


    # train_Y = torch.tensor(train_Y).float().view(-1, 1)
    # train_Y = train_Y.to(DEVICE)
    #
    # test_Y = torch.tensor(test_Y).float().view(-1, 1)
    # test_Y = test_Y.to(DEVICE)
    #
    # probs_train_Y = torch.softmax(model(train_bg), 1)
    # sampled_train_Y = torch.multinomial(probs_train_Y, 1)
    # argmax_train_Y = torch.max(probs_train_Y, 1)[1].view(-1, 1)
    #
    # probs_test_Y = torch.softmax(model(test_bg), 1)
    # sampled_test_Y = torch.multinomial(probs_test_Y, 1)
    # argmax_test_Y = torch.max(probs_test_Y, 1)[1].view(-1, 1)




# Training Faze:
model, testset, trainset = train_model(experiment='countryVSurban')

torch.save(model.state_dict(), 'trained_graph')
# -------------------------------------------------------------
torch.cuda.empty_cache()

# Evaluation Faze:
model = VisualGenomeGN()
model.load_state_dict(torch.load('trained_graph'))
model.to(DEVICE)
model_eval(model, testset, trainset)
# --------------------------------------------------------------
