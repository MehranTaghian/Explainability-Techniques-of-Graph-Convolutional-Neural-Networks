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


def train_model(epoches=100, experiment='countryVSurban'):
    # Create training and test sets.
    trainset, testset = load_data(experiment=experiment)
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
    return model, testset


def model_eval(model, test_set, train_set):
    model.eval()
    # Convert a list of tuples to two lists
    train_X, train_Y = map(list, zip(*train_set))
    test_X, test_Y = map(list, zip(*test_set))

    train_bg = dgl.batch(train_X)
    test_bg = dgl.batch(test_X)

    train_Y = torch.tensor(train_Y).float().view(-1, 1)
    train_Y = train_Y.to(DEVICE)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    test_Y = test_Y.to(DEVICE)

    probs_train_Y = torch.softmax(model(train_bg), 1)
    sampled_train_Y = torch.multinomial(probs_train_Y, 1)
    argmax_train_Y = torch.max(probs_train_Y, 1)[1].view(-1, 1)

    probs_test_Y = torch.softmax(model(test_bg), 1)
    sampled_test_Y = torch.multinomial(probs_test_Y, 1)
    argmax_test_Y = torch.max(probs_test_Y, 1)[1].view(-1, 1)

    print('Accuracy of sampled predictions on the train set: {:.4f}%'.format(
        (train_Y == sampled_train_Y.float()).sum().item() / len(train_Y) * 100))
    print('Accuracy of argmax predictions on the train set: {:4f}%'.format(
        (train_Y == argmax_train_Y.float()).sum().item() / len(train_Y) * 100))

    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
        (test_Y == sampled_test_Y.float()).sum().item() / len(test_Y) * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_test_Y.float()).sum().item() / len(test_Y) * 100))


# Training Faze:
model, testset = train_model(experiment='countryVSurban')
torch.save(model.state_dict(), 'trained_graph')
# -------------------------------------------------------------
torch.cuda.empty_cache()

# Evaluation Faze:
model = Classifier()
model.load_state_dict(torch.load('trained_graph'))
model.to(DEVICE)
model_eval(model, testset)
# --------------------------------------------------------------
