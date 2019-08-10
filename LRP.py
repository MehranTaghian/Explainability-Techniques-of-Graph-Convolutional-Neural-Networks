import torch
from TrainGraphModel import train_model


def lrp(model, epsilon=1e-5):
    params = list(model.parameters())
    R = model.Z[-1]
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
    return R


# model = train_model()
# from GraphDataset import GraphDataset
#
# samples = GraphDataset('countryVSurban')
# model(samples[0][0])
#
# for e in [1e-3, 1e-4, 5e-5, 1e-6]:
#     r = lrp(model, e)
#     print(e)
#     _, index = torch.sort(torch.sum(r, dim=1))
#     print(index)

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
