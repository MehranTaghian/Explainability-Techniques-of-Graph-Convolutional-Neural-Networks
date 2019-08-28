import os
from visual_genome import local as vg
import re
import csv
import pandas as pd

DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome"
TARGET_DIR = DATA_DIR + '\\Edges\\'

classes = ['country', 'urban']  # , 'indoor', 'outdoor']


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
    return graph


def create_edge_files():
    for c in classes:
        image_folder_dir = DATA_DIR + f'\\Regions\\{c}'
        list_image_ids = os.listdir(DATA_DIR + f'\\Regions\\{c}')
        for image_id in list_image_ids:
            print(image_id)
            list_images = os.listdir(image_folder_dir + f'\\{image_id}\\new\\')
            list_images = [l.split('.')[0] for l in list_images]
            graph = vg.get_scene_graph(int(image_id), DATA_DIR + '\\', DATA_DIR + '\\by-id\\',
                                       DATA_DIR + '\\synsets.json')
            # do preprocess ....
            graph = preprocess_object_names(graph)

            edge_set = set()
            for r in graph.relationships:
                sub = r.subject.__str__()
                obj = r.object.__str__()
                sub_ind = list_images.index(sub)
                obj_ind = list_images.index(obj)
                edge_set.add((sub_ind, obj_ind))

            with open(TARGET_DIR + f'{image_id}.txt', mode='w') as edge_file:
                edge_writer = csv.writer(edge_file, delimiter=',')
                edge_writer.writerow(('sub', 'obj'))
                for items in edge_set:
                    edge_writer.writerow(items)


# create_edge_files()


def get_edge_list(image_id):
    return pd.read_csv(TARGET_DIR + f'{image_id}.txt').values

# image_id = 107926
# image_folder_dir = DATA_DIR + f'\\Regions\\country'
# list_images = os.listdir(image_folder_dir + f'\\{image_id}\\new\\')
# list_images = [l.split('.')[0] for l in list_images]
# graph = vg.get_scene_graph(int(image_id), DATA_DIR + '\\', DATA_DIR + '\\by-id\\',
#                            DATA_DIR + '\\synsets.json')
# # do preprocess ....
# graph = preprocess_object_names(graph)
# edge_set = set()
# for r in graph.relationships:
#     sub = r.subject.__str__()
#     obj = r.object.__str__()
#     sub_ind = list_images.index(sub)
#     obj_ind = list_images.index(obj)
#     edge_set.add((sub_ind, obj_ind))
#     print(edge_set)
# # os.mkdir(TARGET_DIR + f'test\\')
# with open(TARGET_DIR + f'test\\{image_id}.txt', mode='w') as edge_file:
#     edge_writer = csv.writer(edge_file, delimiter=',')
#     edge_writer.writerow(('sub', 'obj'))
#     for items in edge_set:
#         edge_writer.writerow(items)
