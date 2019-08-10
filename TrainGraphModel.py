import dgl
import torch
from GraphDataset import GraphDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Graph.GraphModel import Classifier
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(train_percent=80, experiment='countryVSurban'):
    samples = GraphDataset(experiment)
    return train_test_split(samples, train_size=train_percent, shuffle=True)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, label = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(label).to(DEVICE)


def train_model(epoches=80):
    # Create training and test sets.
    trainset, testset = load_data()
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                             collate_fn=collate)

    # g, _, _ = next(iter(data_loader))
    # print(type(g))
    # Create model
    model = Classifier()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(DEVICE)
    model.train()

    epoch_losses = []
    for epoch in range(epoches):
        epoch_loss = 0
        for iter, (bg, labels) in enumerate(data_loader):
            prediction = model(bg)
            loss = loss_func(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    # model.eval()
    # # Convert a list of tuples to two lists
    # test_X, test_Y = map(list, zip(*testset))
    # test_bg = dgl.batch(test_X)
    # print(test_Y)
    # test_Y = torch.tensor(test_Y).float().view(-1, 1)
    # print(test_Y)
    # test_Y = test_Y.to(DEVICE)
    # probs_Y = torch.softmax(model(test_bg), 1)
    # sampled_Y = torch.multinomial(probs_Y, 1)
    # argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    # print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    #     (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    # print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    #     (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
    return model

# model = train_model()
# for i in model.parameters():
#     print(i.data.shape)
# samples = GraphDataset('countryVSurban')
# model(samples[0][0])
# for i in model.output:
#     print(i.shape)
