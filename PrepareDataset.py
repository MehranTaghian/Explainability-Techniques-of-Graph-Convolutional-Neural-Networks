import visual_genome.local as vg
from shutil import copyfile
import numpy as np

DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome\\"
TARGET_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome\ClassifiedImages\\"

# Keywords
country_keywords = ['country', 'countryside', 'farm', 'rural', 'cow', 'crops', 'sheep']
urban_keywords = ['urban', 'city', 'downtown']
indoor_keywords = ['indoor', 'room', 'ofﬁce', 'bedroom', 'bathroom']
outdoor_keywords = ['outdoor', 'nature', 'outside']

graphs = vg.get_scene_graphs(start_index=0, end_index=-1, data_dir=DATA_DIR,
                             image_data_dir=DATA_DIR + 'by-id\\')

# graphs = [vg.get_scene_graph(107926, DATA_DIR,
#                              image_data_dir=DATA_DIR + 'by-id\\', synset_file=DATA_DIR + 'synsets.json'),
#           vg.get_scene_graph(200, DATA_DIR,
#                              image_data_dir=DATA_DIR + 'by-id\\', synset_file=DATA_DIR + 'synsets.json')]

QA = vg.get_all_qas(DATA_DIR)


def get_qas_for_image(id):
    for q in QA:
        if len(q) >= 1:
            if q[0].image.id == id:
                return q
    return None


print("copying...")

for g in graphs:
    which_class = [False, False, False, False]
    for r in g.relationships:
        if any(k in r.__str__() for k in country_keywords) or any(k in r.synset.__str__() for k in country_keywords):
            which_class[0] = True
        if any(k in r.__str__() for k in urban_keywords) or any(k in r.synset.__str__() for k in urban_keywords):
            which_class[1] = True
        if any(k in r.__str__() for k in indoor_keywords) or any(k in r.synset.__str__() for k in indoor_keywords):
            which_class[2] = True
        if any(k in r.__str__() for k in outdoor_keywords) or any(k in r.synset.__str__() for k in outdoor_keywords):
            which_class[3] = True

    temp = np.where(which_class)[0]
    if temp.shape[0] <= 1:
        for a in g.attributes:
            if any(k in a.__str__() for k in country_keywords):
                which_class[0] = True
            if any(k in a.__str__() for k in urban_keywords):
                which_class[1] = True
            if any(k in a.__str__() for k in indoor_keywords):
                which_class[2] = True
            if any(k in a.__str__() for k in outdoor_keywords):
                which_class[3] = True

    temp = np.where(which_class)[0]
    if temp.shape[0] <= 1:
        for o in g.objects:
            if any(k in o.synsets.__str__() for k in country_keywords):
                which_class[0] = True
            if any(k in o.synsets.__str__() for k in urban_keywords):
                which_class[1] = True
            if any(k in o.synsets.__str__() for k in indoor_keywords):
                which_class[2] = True
            if any(k in o.synsets.__str__() for k in outdoor_keywords):
                which_class[3] = True

    temp = np.where(which_class)[0]
    if temp.shape[0] <= 1:
        qas = get_qas_for_image(g.image.id)
        if qas is not None:
            for q in qas:
                if any(k in q.question.__str__().lower() for k in country_keywords):
                    which_class[0] = True
                if any(k in q.question.__str__().lower() for k in urban_keywords):
                    which_class[1] = True
                if any(k in q.question.__str__().lower() for k in indoor_keywords):
                    which_class[2] = True
                if any(k in q.question.__str__().lower() for k in outdoor_keywords):
                    which_class[3] = True

    temp = np.where(which_class)[0]

    if temp.shape[0] == 1:
        if temp == 0:  # country
            copyfile(DATA_DIR + "images\\" + str(g.image.id) + ".jpg",
                     TARGET_DIR + "country\\" + str(g.image.id) + ".jpg")
        elif temp == 1:  #
            copyfile(DATA_DIR + "images\\" + str(g.image.id) + ".jpg",
                     TARGET_DIR + "urban\\" + str(g.image.id) + ".jpg")

        elif temp == 2:  # indoor
            copyfile(DATA_DIR + "images\\" + str(g.image.id) + ".jpg",
                     TARGET_DIR + "indoor\\" + str(g.image.id) + ".jpg")

        elif temp == 3:  # outdoor
            copyfile(DATA_DIR + "images\\" + str(g.image.id) + ".jpg",
                     TARGET_DIR + "outdoor\\" + str(g.image.id) + ".jpg")
