import torch
from Graph.GraphModel import Classifier
from CreateGraphEdge import get_edge_list
import dgl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pil import Image as PIL_Image
from visual_genome import local as vg
import os
import numpy as np
from sklearn.preprocessing import normalize


def lrp(model, epsilon=1e-5):
    params = list(model.parameters())
    R = model.Z[-1]
    temp = params[-2]
    temp2 = params[-1]
    max_index = torch.argmax(R)
    params[-2] = torch.zeros(params[-2].shape, device='cuda:0')
    params[-1] = torch.zeros(params[-1].shape, device='cuda:0')
    params[-2][max_index, :] = temp[max_index, :]
    params[-1][max_index] = temp2[max_index]
    X = []
    for l in model.layers:
        X.append(l.X)
    X.append(model.X)
    i = len(model.Z) - 1
    while i >= 0:
        Z = model.Z[i]
        if len(X[i].shape) > 1:
            partial_z = X[i][torch.arange(X[i].shape[0]), :, None] * params[2 * i].data.t()
        else:
            partial_z = X[i][:, None] * params[2 * i].data.t()

        partial_z += params[2 * i + 1]
        temp = torch.tensor(Z >= 0, dtype=torch.float).to('cuda:0')
        temp += torch.tensor(Z < 0, dtype=torch.float).to('cuda:0') * -1
        Z += epsilon * temp
        Z = 1 / Z
        R = Z * R
        # print(R.shape)
        # print(partial_z.shape)
        if len(X[i].shape) > 1:
            R = partial_z * R[:, None, :]
            R = torch.sum(R, dim=2)
        else:
            R = partial_z * R
            R = torch.sum(R, dim=1)
        i -= 1
    R = torch.sum(R, dim=1)
    # R = R.cpu().detach().numpy()
    # R = normalize(R[:, np.newaxis], axis=0).ravel()
    return R


DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome"
NODES_DIR = DATA_DIR + '\\Features'
EDGES_DIR = DATA_DIR + '\\Edges'
REGIONS_DIR = DATA_DIR + '\\Regions'


def get_sample(image_id, label, experiment='countryVSurban'):
    features = torch.load(NODES_DIR + F'\\{experiment}\\node-features-{image_id}-{label}.pt')
    edge = get_edge_list(image_id)
    g = dgl.DGLGraph()
    g.add_nodes(features.shape[0])
    g.add_edges(u=edge[:, 0], v=edge[:, 1])
    g.ndata['h'] = features
    return g, label


def draw_sample(image_id, label, relevance_sorted_indices, sample_id, experiment='countryVSurban'):
    graph = vg.get_scene_graph(image_id, DATA_DIR, DATA_DIR + '\\by-id\\', DATA_DIR + '\\synsets.json')
    objects = graph.objects
    categ = None
    if experiment == 'countryVSurban':
        categ = 'country' if label == 0 else 'urban'
    else:
        categ = 'indoor' if label == 0 else 'outdoor'

    img = PIL_Image.open(DATA_DIR + f'\\images\\{image_id}.jpg')
    plt.imshow(img)
    ax = plt.gca()

    list_regions = os.listdir(REGIONS_DIR + f'\\{categ}\\{image_id}\\new')
    relevance_sorted_indices = relevance_sorted_indices.cpu().detach().numpy()

    j = 1
    for r in relevance_sorted_indices[:len(relevance_sorted_indices) - 7:-1]:
        for o in objects:
            if list_regions[r].split('.')[0] in o.__str__():
                ax.add_patch(Rectangle((o.x, o.y),
                                       o.width,
                                       o.height,
                                       fill=False,
                                       edgecolor='green',
                                       linewidth=3))
                ax.text(o.x, o.y, str(j) + o.__str__(), style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
                j += 1
                break
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    # plt.show()
    plt.savefig(f'{sample_id}-{categ}.png')
    plt.close()

def random_picker(sample_id, experiment='countryVSurban'):
    list_samples = os.listdir(os.path.join(NODES_DIR, experiment))
    sample_name = list_samples[np.random.randint(len(list_samples))]
    image_id = int(sample_name.split('-')[2])
    print(image_id)
    label = int((sample_name.split('-')[3]).split('.')[0])
    sample = get_sample(image_id, label, experiment)
    model(sample[0])
    values, sorted_relevance = torch.sort(lrp(model))
    draw_sample(image_id, label, sorted_relevance, sample_id, experiment)


model = Classifier()
model.load_state_dict(torch.load('trained_graph'))
model.to('cuda:0')
for i in range(1, 20):
    print(i)
    random_picker(i)

# def lrp(input, model, number_of_layers=5):
#     params = list(model.parameters())
#     layer = [input, torch.mm(input, params[0].data.t()) + params[1].data]
#     param_index = 2
#     for i in range(2, number_of_layers):
#         layer.append(torch.mm(layer[i - 1], params[param_index].data.t()) + params[param_index + 1].data)
#         param_index += 2
#
#     x = number_of_layers - 1
#     R_j = layer[x]  # last layer
#     x -= 1
#     param_index = len(params) - 1
#     while x >= 0:
#         W = params[param_index - 1].data.t()
#         Z = layer[x][:, None] * W
#         b = params[param_index].data
#         Z += b
#         R_jtoi = (Z / layer[x + 1][:, None]) * R_j  # (Z_i,j / Z_j) * R_j
#         print(R_jtoi.shape)
#         R_j = torch.sum(R_jtoi, dim=1)
#         x -= 1
#     return R_j


# def lrp(model, epsilon=1e-5):
#     params = list(model.parameters())
#     R = model.Z[-1]
#     X = []
#     for l in model.layers:
#         X.append(l.X)
#     X.append(model.X)
#     i = len(model.Z) - 1
#     while i >= 0:
#         Z = model.Z[i]
#         if len(X[i].shape) > 1:
#             partial_z = X[i][torch.arange(X[i].shape[0]), :, None] * params[2 * i].data.t()
#         else:
#             partial_z = X[i][:, None] * params[2 * i].data.t()
#
#         partial_z += params[2 * i + 1]
#         temp = torch.tensor(Z >= 0, dtype=torch.float).to('cuda:0')
#         temp += torch.tensor(Z < 0, dtype=torch.float).to('cuda:0') * -1
#         Z += epsilon * temp
#         Z = 1 / Z
#         R = Z * R
#         # print(R.shape)
#         # print(partial_z.shape)
#         if len(X[i].shape) > 1:
#             R = partial_z * R[:, None, :]
#             R = torch.sum(R, dim=2)
#         else:
#             R = partial_z * R
#             R = torch.sum(R, dim=1)
#         i -= 1
#     return R
