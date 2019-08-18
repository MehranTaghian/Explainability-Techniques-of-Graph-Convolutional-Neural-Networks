import visual_genome.local as vg
from pil import Image as PIL_Image
import os
import re

DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome"
TARGET_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome\ClassifiedImages"
REGION_DIR = "C:\\Users\\Mehran\\Desktop\\Azizpour\\Datasets\\Gnome\\Regions"

image_data_dir = DATA_DIR + '\\by-id\\'
image_dir = DATA_DIR + "\\images\\"
Categories = ['country', 'urban', 'indoor', 'outdoor']


def crop_regions(image_id, objects, category):
    """
    Gets the id of image and selected regions and crops the regions for input to the InceptionV3
    :param category: It is "urban" / "country" / "indoor" / "outdoor"
    :param image_id:
    :param regions:
    :return:
    """
    i = 0
    for o in objects:
        w = o.width
        h = o.height
        x = o.x
        y = o.y
        img = PIL_Image.open(image_dir + str(image_id) + ".jpg")
        cropped = img.crop((x, y, x + w, y + h))
        if not os.path.exists(
                REGION_DIR + f"\\{category}\\{image_id}\\new"):
            os.makedirs(REGION_DIR + f"\\{category}\\{image_id}\\new")
        file_name = o.__str__()
        # file_name = re.sub('[\W_]+', ' ', file_name).strip()
        # if file_name != o.names[0]:
        #     print(i)
        #     print(file_name)
        #     print(o.names[0])
        if len(file_name) > 0:
            file_path = REGION_DIR + f"\\{category}\\{image_id}\\new\\{file_name}.jpg"
            cropped.save(file_path)
        else:
            print(i)
            print(o.names[0])
        i += 1
    print(f"Cropped regions saved for image {image_id}")


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

    return object_list


def get_objects_of_graph():
    for c in Categories:
        list_images = os.listdir(TARGET_DIR + f"\\{c}")
        for image in list_images:
            image_id = int(image.split('.')[0])
            print(f'Saving for {image_id}...')
            graph = vg.get_scene_graph(image_id, DATA_DIR, image_data_dir, DATA_DIR + '\\synsets.json')
            list_objects = preprocess_object_names(graph)
            crop_regions(image_id, list_objects, c)


# graph = vg.get_scene_graph(107926, DATA_DIR, image_data_dir, DATA_DIR + '\\synsets.json')
# objects_list = preprocess_object_names(graph)
# crop_regions(107926, objects_list, 'country')

get_objects_of_graph()
