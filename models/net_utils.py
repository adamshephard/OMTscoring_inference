import torch.nn as nn

def convert_state_dict(state_dict):
    new_state_dict = state_dict.copy()
    for k in state_dict.keys():
        if 'module' in k:
            k2 = k.split('module.')[1]
            new_state_dict[k2] = new_state_dict.pop(k)
    return new_state_dict

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x