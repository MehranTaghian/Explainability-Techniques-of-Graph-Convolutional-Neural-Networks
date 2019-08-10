import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import visual_genome.local as vg
from pil import Image as PIL_Image
import os
import re

DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome"
TARGET_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome\ClassifiedImages"
image_data_dir = DATA_DIR + '\\by-id\\'
image_dir = DATA_DIR + "\\images\\"
Categories = ['country', 'urban', 'indoor', 'outdoor']


def visualize_regions(image_dir, regions):
    img = PIL_Image.open(image_dir)
    plt.imshow(img)
    ax = plt.gca()
    #     with regions[0] as region:
    i = 1
    for region in regions[0:10]:
        ax.add_patch(Rectangle((region.x, region.y),
                               region.width,
                               region.height,
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(region.x, region.y, str(i) + region.phrase, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
        i += 1
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()


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
                f"C:\\Users\\Mehran\\Desktop\\Azizpour\\Datasets\\Gnome\\Regions\\{category}\\" + str(image_id)):
            os.makedirs(f"C:\\Users\\Mehran\\Desktop\\Azizpour\\Datasets\\Gnome\\Regions\\{category}\\" + str(image_id))
        file_name = o.names[0]
        file_name = re.sub('[\W_]+', ' ', file_name).strip()
        if file_name != o.names[0]:
            print(i)
            print(file_name)
            print(o.names[0])
        if len(file_name) > 0:
            cropped.save(
                f"C:\\Users\\Mehran\\Desktop\\Azizpour\\Datasets\\Gnome\\Regions\\{category}\\" + str(
                    image_id) + "\\" + file_name + ".jpg")
        else:
            print(i)
            print(o.names[0])
        i += 1
    print(f"Cropped regions saved for image {image_id}")


def get_objects_of_graph():
    for c in Categories:
        list_images = os.listdir(TARGET_DIR + f"\\{c}")
        for image in list_images:
            image_id = int(image.split('.')[0])
            print(f'Saving for {image_id}...')
            graph = vg.get_scene_graph(image_id, DATA_DIR, image_data_dir, DATA_DIR + '\\synsets.json')
            crop_regions(image_id, graph.objects, c)


get_objects_of_graph()
