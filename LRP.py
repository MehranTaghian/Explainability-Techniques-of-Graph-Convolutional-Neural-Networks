from matplotlib.patches import Circle
import matplotlib.cbook as cbook
from CreateGraphEdge import get_edge_list
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image
from visual_genome import local as vg
from CreateGraphEdge import preprocess_object_names
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import dgl.function as fn
import os
import re
import torch
import dgl

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    # accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}


def lrp(model, input, image_id, epsilon=1e-5):
    params = list(model.parameters())
    R = model.Z[-1]
    temp = params[-2]
    temp2 = params[-1]
    max_index = torch.argmax(R)
    params[-2] = torch.zeros(params[-2].shape, device='cuda:0')
    params[-1] = torch.zeros(params[-1].shape, device='cuda:0')
    params[-2][max_index, :] = temp[max_index, :]
    params[-1][max_index] = temp2[max_index]
    X = [input]
    # for l in range(len(model.layers) - 1):
    for l in range(len(model.layers)):
        print(model.layers[l].apply_mod.X.shape)
        X.append(model.layers[l].apply_mod.X)
        # X.append(l.X)
    g = dgl.DGLGraph()
    g.add_nodes(input.shape[0])
    edge = get_edge_list(image_id)
    g.add_edges(u=edge[:, 0], v=edge[:, 1])
    # X.append(model.X)
    i = len(model.Z) - 1
    level = 0
    while i >= 0:
        print(level)
        level += 1
        Z = model.Z[i]
        # print('X', X[i].shape)
        # print('params', params[2 * i].shape)
        if len(X[i].shape) > 1:
            # partial_z = X[i][torch.arange(X[i].shape[0]), :, None] * params[2 * i].data.t()
            partial_z = X[i][torch.arange(X[i].shape[0]), :, None] * params[2 * i].data.t()
        else:
            partial_z = X[i][:, None] * params[2 * i].data.t()

        partial_z += params[2 * i + 1]
        temp = torch.tensor(Z >= 0, dtype=torch.float).to('cuda:0')
        temp += torch.tensor(Z < 0, dtype=torch.float).to('cuda:0') * -1
        Z += epsilon * temp
        print('Z', Z.shape)
        print('partial_Z', partial_z.shape)
        print('R.shape', R.shape)
        Z = 1 / Z
        R = Z * R
        if len(X[i].shape) > 1 and level > 1:
            R = partial_z * R[:, None, :]
            R = torch.sum(R, dim=2)
        else:
            R = partial_z * R
            R = torch.sum(R, dim=2)
        # print(torch.sum(R))
        # g.ndata['h'] = R
        # g.update_all(msg, reduce)
        # R = g.ndata['h']
        print('R', R)
        print(f'sum:{torch.sum(R)}')
        i -= 1
    R = torch.mean(R, dim=1)
    # R = torch.sum(R, dim=1)
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


def preprocess_object_names(graph):
    for i in range(len(graph.relationships)):
        graph.relationships[i].subject.names[0] = re.sub('[\W_]+', ' ', graph.relationships[i].subject.names[0]).strip()
        graph.relationships[i].object.names[0] = re.sub('[\W_]+', ' ', graph.relationships[i].object.names[0]).strip()

    rel = graph.relationships

    for i in range(len(rel)):
        counter = 1
        name = rel[i].subject.__str__()
        for j in range(i, len(rel)):
            if j == i:
                name_second = rel[j].object.__str__()
                if name == name_second:
                    graph.relationships[j].object.names[0] += str(counter)
                    counter += 1
            else:
                name_second = rel[j].subject.__str__()
                if name == name_second:
                    graph.relationships[j].subject.names[0] += str(counter)
                    counter += 1

                name_second = rel[j].object.__str__()
                if name == name_second:
                    graph.relationships[j].object.names[0] += str(counter)
                    counter += 1
    object_list_names = []
    object_list = []
    for r in graph.relationships:
        if r.subject.__str__() not in object_list_names:
            object_list.append(r.subject)
        if r.object.__str__() not in object_list_names:
            object_list.append(r.object)

    return object_list, graph


def draw_sample_v3(sample_id, image_id, label, relevance_sorted_indices, relevance_values,
                   experiment='countryVSurban'):
    relevance_sorted_indices = relevance_sorted_indices.cpu().detach().numpy()
    graph = vg.get_scene_graph(image_id, DATA_DIR, DATA_DIR + '\\by-id\\', DATA_DIR + '\\synsets.json')
    objects, graph = preprocess_object_names(graph)

    if experiment == 'countryVSurban':
        categ = 'country' if label == 0 else 'urban'
    else:
        categ = 'indoor' if label == 0 else 'outdoor'
    objects_dir = REGIONS_DIR + f'\\{categ}\\{image_id}\\new'
    list_regions = os.listdir(objects_dir)

    radius_list = []
    r = 40
    top5_objects = []
    for top_relevance in relevance_sorted_indices[:len(relevance_sorted_indices) - 6:-1]:
        radius_list += [r]
        r -= 8

        for o in objects:
            if o.__str__() == list_regions[top_relevance].split('.')[0]:
                top5_objects += [o]
                break

    top5_objects_dict = {}
    for o in top5_objects:
        top5_objects_dict[o.__str__()] = o

    edges = []
    for r in graph.relationships:
        for o1 in top5_objects:
            if r.subject.__str__() == o1.__str__():
                for o2 in top5_objects:
                    if r.object.__str__() == o2.__str__():
                        edges += [r]

    image_file = cbook.get_sample_data(DATA_DIR + f'\\images\\{image_id}.jpg')
    img = plt.imread(image_file)

    # Make some example data
    # x = np.random.rand(5)*img.shape[1]
    # y = np.random.rand(5)*img.shape[0]

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    # Show the image
    ax.imshow(img)

    # Now, loop through coord arrays, and create a circle at each x,y pair
    r = 0
    i = -1
    for o in top5_objects:
        color = 'green' if relevance_values[i] >= 0 else 'red'
        circ = Circle((o.x + o.width / 2, o.y + o.height / 2), radius_list[r], color=color)
        i -= 1
        r += 1
        ax.add_patch(circ)
        ax.text(o.x + o.width / 2, o.y + o.height / 2, o.__str__(), bbox=dict(facecolor='white', alpha=0.7))

    for e in edges:
        subject = top5_objects_dict[e.subject.__str__()]
        object = top5_objects_dict[e.object.__str__()]
        x = [subject.x + subject.width / 2, object.x + object.width / 2]
        y = [subject.y + subject.height / 2, object.y + object.height / 2]
        # print(f'{e.subject}:{(e.subject.x + e.subject.width / 2, e.subject.y + e.subject.height / 2)}')
        # print(f'{e.object}:{(e.object.x + e.object.width / 2, e.object.y + e.object.height / 2)}')
        plt.plot(x, y, 'b', linewidth=3)
        plt.text(abs(x[0] + x[1]) / 2, abs(y[0] + y[1]) / 2, e.predicate.__str__(),
                 bbox=dict(facecolor='white', alpha=0.7))
    # Show the image
    plt.savefig(f'Result\\{sample_id}-{categ}-{image_id}.png')


def draw_sample_v2(sample_graph, image_id, label, relevance_sorted_indices, sample_id, relevance_values,
                   experiment='countryVSurban'):
    relevance_sorted_indices = relevance_sorted_indices.cpu().detach().numpy()

    graph = vg.get_scene_graph(image_id, DATA_DIR, DATA_DIR + '\\by-id\\', DATA_DIR + '\\synsets.json')
    objects, graph = preprocess_object_names(graph)
    G = nx.Graph()
    if experiment == 'countryVSurban':
        categ = 'country' if label == 0 else 'urban'
    else:
        categ = 'indoor' if label == 0 else 'outdoor'

    objects_dir = REGIONS_DIR + f'\\{categ}\\{image_id}\\new'
    list_regions = os.listdir(objects_dir)
    for i in range(len(sample_graph.nodes)):
        img = mpimg.imread(f'{objects_dir}\\{list_regions[i]}')
        G.add_node(i, image=img)

    nodes_dic = {}
    i = 0
    for f in list_regions:
        temp = f.split('.')[0]
        nodes_dic[temp] = i
        i += 1

    edge_labels = {}
    for r in graph.relationships:
        edge = (nodes_dic[r.subject.__str__()], nodes_dic[r.object.__str__()])
        G.add_edge(edge[0], edge[1])
        edge_labels[edge] = r.predicate

    pos = nx.planar_layout(G, 4)

    piesize = np.ones(len(pos)) * 0.002
    index_image = 0.001
    index_image_increment = 0.001
    index = 0.005
    index_increment = 0.005

    for i in relevance_sorted_indices:
        pos[i] += index
        piesize[i] += index_image
        index += index_increment
        index_image += index_image_increment

    fig = plt.figure(figsize=(100, 100))
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    i = 0
    for n in G:
        p2 = piesize[i] / 2
        xx, yy = trans(pos[n])  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        a = plt.axes([xa - p2, ya - p2, piesize[i], piesize[i]])
        a.set_aspect('equal')
        a.imshow(G.node[n]['image'])
        a.axis('off')
        i += 1

    ax.axis('off')
    plt.savefig(f'graph-{sample_id}-{categ}.png')
    plt.close()

    img = PIL_Image.open(DATA_DIR + f'\\images\\{image_id}.jpg')
    plt.imshow(img)
    ax = plt.gca()

    j = 1
    for i in range(len(relevance_sorted_indices)):
        for o in objects:
            if list_regions[relevance_sorted_indices[i]].split('.')[0] in o.__str__():
                if relevance_values[i] < 0:
                    ax.add_patch(Rectangle((o.x, o.y),
                                           o.width,
                                           o.height,
                                           fill=False,
                                           edgecolor='red',
                                           linewidth=3))
                else:
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
        i += 1
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    # plt.show()
    plt.savefig(f'{sample_id}-{categ}.png')
    plt.close()


def draw_sample(image_id, label, relevance_sorted_indices, sample_id, relevance_values, experiment='countryVSurban'):
    graph = vg.get_scene_graph(image_id, DATA_DIR, DATA_DIR + 'by-id/', DATA_DIR + '/synsets.json')
    objects = preprocess_object_names(graph)

    categ = None
    if experiment == 'countryVSurban':
        categ = 'country' if label == 0 else 'urban'
    else:
        categ = 'indoor' if label == 0 else 'outdoor'

    img = PIL_Image.open(DATA_DIR + f'images/{image_id}.jpg')
    plt.imshow(img)
    ax = plt.gca()

    list_regions = os.listdir(REGIONS_DIR + f'{categ}/{image_id}/new')
    relevance_sorted_indices = relevance_sorted_indices.cpu().detach().numpy()

    j = 1
    for i in range(relevance_sorted_indices):
        for o in objects:
            if list_regions[relevance_sorted_indices[i]].split('.')[0] in o.__str__():
                if relevance_values[i] < 0:
                    ax.add_patch(Rectangle((o.x, o.y),
                                           o.width,
                                           o.height,
                                           fill=False,
                                           edgecolor='red',
                                           linewidth=3))
                else:
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
        i += 1
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    # plt.show()
    plt.savefig(f'{sample_id}-{categ}.png')
    plt.close()


def random_picker(model, sample_id, experiment='countryVSurban'):
    list_samples = os.listdir(os.path.join(NODES_DIR, experiment))
    sample_name = list_samples[np.random.randint(len(list_samples))]
    image_id = int(sample_name.split('-')[2])
    print(image_id)
    label = int((sample_name.split('-')[3]).split('.')[0])
    sample = get_sample(image_id, label, experiment)
    temp = sample[0].ndata['h']
    predict = model(sample[0])
    label_predicted = torch.argmax(predict)
    print(label_predicted)
    if label_predicted == label:
        print("Predicted")
    else:
        print("False")
    values, sorted_relevance = torch.sort(lrp(model, temp, image_id))
    print(values)
    draw_sample(sample[0], image_id, label, sorted_relevance, sample_id, experiment)


# image_id = 107926
# label = 0
# model = Classifier()
# model.load_state_dict(torch.load('Graph Model Trained\\trained_graph'))
# model.to('cuda:0')
# sample = get_sample(image_id, label, 'countryVSurban')
# temp = sample[0].ndata['h']
# model(sample[0])
# # lrp(model, temp)
# values, sorted_relevance = torch.sort(lrp(model, temp, image_id))
# print(values)
# draw_sample_v2(sample[0], image_id, label, sorted_relevance, 1, values)
# draw_sample_v3(1, image_id, label, sorted_relevance, values,
#                experiment='countryVSurban')

# for i in range(1, 20):
#     print(i)
#     random_picker(i)

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
