from GraphDataset import GraphDataset
from Graph.VisualGenomeGNRelevance import VisualGenomeGNRelevance
from Federico.torchgraphs.src import torchgraphs as tg
import torch
import numpy as np

net_lrp = VisualGenomeGNRelevance('avg', True)
net_lrp.load_state_dict(torch.load(r'..\\trained_graph', map_location='cuda:0'))
net_lrp.to('cuda:0')

TARGET_FEATURES_DIR = r"C:\Users\Mehran\Desktop\Azizpour\Datasets\Gnome\Features\Fidelity\\"


def fidelity_LRP_for_class(class_label, experiment):
    """

    :param class_label: For calculating fidelity for each class, the score for that class should be backpropagated
                        in each method of explainability
    :param experiment: "country VS urban" or "indoor VS outdoor"
    :return:
    """
    global target_fidelity_class
    if experiment == 'countryVSurban':
        target_fidelity_class = 'country' if class_label == 0 else 'urban'
    elif experiment == 'indoorVSoutdoor':
        target_fidelity_class = 'indoor' if class_label == 0 else 'outdoor'

    datas = GraphDataset(experiment)
    for image_id, (graph_in, label) in zip(datas.ids, datas.sample):
        print(image_id)
        batch = tg.GraphBatch.collate([graph_in]).requires_grad_()
        graph_out = net_lrp(batch)[0]
        global_relevance = torch.zeros_like(graph_out.global_features)
        global_relevance[class_label] = graph_out.global_features[class_label]
        graph_in.zero_grad_()
        graph_out.global_features.backward(global_relevance)
        saliency = batch.node_features.grad.sum(dim=1)
        saliency = saliency.detach().cpu().numpy()
        features = graph_in.node_features.cpu()
        features[np.where(saliency > 0), :] = torch.zeros(2048)
        torch.save(features.to('cuda:0'),
                   TARGET_FEATURES_DIR + f'{target_fidelity_class}\\node-features-{image_id}-{label}.pt')


fidelity_LRP_for_class(0, 'countryVSurban')
fidelity_LRP_for_class(1, 'countryVSurban')


class QuantitativeMetrics:

    def get_binarized_hitmaps(self, graph_in, class_label):
        batch = tg.GraphBatch.collate([graph_in]).requires_grad_()
        graph_out = net_lrp(batch)[0]
        global_relevance = torch.zeros_like(graph_out.global_features)
        global_relevance[class_label] = graph_out.global_features[class_label]
        graph_in.zero_grad_()
        graph_out.global_features.backward(global_relevance)
        saliency = batch.node_features.grad.sum(dim=1)
        saliency = saliency.detach().cpu().numpy()
        indices_1 = np.where(saliency >= 0)
        indices_0 = np.where(saliency < 0)
        saliency[indices_1] = 1
        saliency[indices_0] = 0
        return saliency.astype(int).tolist()

    def get_sparsity(self, experiment):
        """

        :param class_label: For calculating fidelity for each class, the score for that class should be backpropagated
                            in each method of explainability
        :param experiment: "country VS urban" or "indoor VS outdoor"
        :return:
        """

        overal_m0 = []
        overal_m1 = []
        datas = GraphDataset(experiment)
        for (graph_in, label) in datas.sample:
            overal_m0 += self.get_binarized_hitmaps(graph_in, 0)
            overal_m1 += self.get_binarized_hitmaps(graph_in, 1)
        overal_m0 = np.array(overal_m0)
        overal_m1 = np.array(overal_m1)
        number_of_non_identified_objects = 0
        for i in range(overal_m0.shape[0]):
            if overal_m0[i] == overal_m1[i] and overal_m0[i] == 0:
                number_of_non_identified_objects += 1
        # (m0 V m1) means that : (overal_m0.shape[0] - number_of_non_identified_objects)
        print(f'Sparsity value: {1 - (overal_m0.shape[0] - number_of_non_identified_objects) / overal_m0.shape[0]}')

    def get_contrastivity(self, experiment):
        """

        :param class_label: For calculating fidelity for each class, the score for that class should be backpropagated
                            in each method of explainability
        :param experiment: "country VS urban" or "indoor VS outdoor"
        :return:
        """

        overal_m0 = []
        overal_m1 = []
        datas = GraphDataset(experiment)
        i = 0
        for (graph_in, label) in datas.sample:
            print(i)
            i += 1
            overal_m0 += self.get_binarized_hitmaps(graph_in, 0)
            overal_m1 += self.get_binarized_hitmaps(graph_in, 1)
        overal_m0 = np.array(overal_m0)
        overal_m1 = np.array(overal_m1)
        hamming_distance = np.sum(overal_m1 != overal_m0)
        number_of_non_identified_objects = 0
        for i in range(overal_m0.shape[0]):
            if overal_m0[i] == overal_m1[i] and overal_m0[i] == 0:
                number_of_non_identified_objects += 1
        # (m0 V m1) means that : (overal_m0.shape[0] - number_of_non_identified_objects)
        print(f'Contrativity value: {hamming_distance / (overal_m0.shape[0] - number_of_non_identified_objects)}')
