import os

import matplotlib.pyplot as plt
import torch
from ImageDataLoader import ImageRegionData
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

DATA_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome"
image_data_dir = DATA_DIR + '\\by-id\\'
image_dir = DATA_DIR + "\\images\\"
classes = ['country', 'urban']

TARGET_FEATURES_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome\Features"

fig = plt.gcf()
fig.set_size_inches(100, 100)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def freeze_all(model_params):
    for param in model_params:
        param.requires_grad = False


def get_model():
    model = models.inception_v3(pretrained=True)
    fc = nn.Linear(2048, 2048)
    fc.weight = nn.Parameter(torch.eye(2048))
    fc.bias = nn.Parameter(torch.zeros(2048))
    model.fc = fc
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(DEVICE)
    return model


def get_inception_features(image_id, category):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_data = ImageRegionData(DATA_DIR + f"\\Regions\\{category}\\{image_id}\\", transform)
    batch_size = len(image_data)
    # print(image_data[0])
    loader = DataLoader(image_data, batch_size=batch_size)
    model = get_model()
    model.eval()
    train = next(iter(loader))[0]
    train = train.to(DEVICE)
    return model.forward(train)


def save_inception_features(class_items):
    i = 0
    target_experiment = 'countryVSurban' if ('country' in class_items) else 'indoorVSoutdoor'
    for c in class_items:
        list_folders = os.listdir(DATA_DIR + f'\\Regions\\{c}')
        for f in list_folders:
            torch.save(get_inception_features(f, c),
                       TARGET_FEATURES_DIR + f'\\{target_experiment}\\node-features-{f}-{i}.pt')

        i += 1  # now second class


experiment1 = ['country', 'urban']
experiment2 = ['indoor', 'outdoor']

save_inception_features(experiment1)
