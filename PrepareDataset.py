import visual_genome.local as vg
from shutil import copyfile
import numpy as np

DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome"
TARGET_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome\ClassifiedImages"

# Keywords
country_keywords = ['country', 'countryside', 'farm', 'rural', 'cow', 'crops', 'sheep']
urban_keywords = ['urban', 'city', 'downtown']
indoor_keywords = ['indoor', 'room', 'ofﬁce', 'bedroom', 'bathroom']
outdoor_keywords = ['outdoor', 'nature', 'outside']

graphs = vg.get_scene_graphs(start_index=0, end_index=-1, data_dir=DATA_DIR + '\\',
                             image_data_dir=DATA_DIR + '\\by-id\\')

counter = np.ones(4, dtype=int)

print("copying...")

for g in graphs:
    which_class = [False, False, False, False]
    for r in g.relationships:
        if any(k in r.__str__() for k in country_keywords):
            which_class[0] = True
        if any(k in r.__str__() for k in urban_keywords):
            which_class[1] = True
        if any(k in r.__str__() for k in indoor_keywords):
            which_class[2] = True
        if any(k in r.__str__() for k in outdoor_keywords):
            which_class[3] = True

    temp = np.where(which_class)[0]
    if temp.shape[0] == 1:
        if temp == 0:  # country
            copyfile(DATA_DIR + "\\images\\" + str(g.image.id) + ".jpg",
                     TARGET_DIR + "\\country\\" + str(g.image.id) + ".jpg")
            counter[0] += 1
        elif temp == 1:  # urban
            copyfile(DATA_DIR + "\\images\\" + str(g.image.id) + ".jpg",
                     TARGET_DIR + "\\urban\\" + str(g.image.id) + ".jpg")
            counter[1] += 1

        elif temp == 2:  # indoor
            copyfile(DATA_DIR + "\\images\\" + str(g.image.id) + ".jpg",
                     TARGET_DIR + "\\indoor\\" + str(g.image.id) + ".jpg")
            counter[2] += 1

        elif temp == 3:  # outdoor
            copyfile(DATA_DIR + "\\images\\" + str(g.image.id) + ".jpg",
                     TARGET_DIR + "\\outdoor\\" + str(g.image.id) + ".jpg")
            counter[3] += 1
