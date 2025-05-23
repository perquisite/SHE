import torch.nn as nn
import torch.nn.functional as F
import torch
import cal_utils as utils
from torch.autograd import Variable
import math
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import copy

class Prototype_embedding(torch.nn.Module):
    def __init__(self, num_prototypes, binary_bits):
            super(Prototype_embedding, self).__init__()
            mid_num1 = 4096
            mid_num2 = 4096
            self.fc1 = nn.Linear(num_prototypes, mid_num1)
            self.fc2 = nn.Linear(mid_num1, mid_num2)
            self.Embedding = nn.Linear(mid_num2, binary_bits)
            nn.init.uniform_( self.Embedding.weight, -1. / np.sqrt(np.float32(num_prototypes)), 1. / np.sqrt(np.float32(num_prototypes)))

    def forward(self, x):
            out1 = F.relu(self.fc1(x))
            out2 = F.relu(self.fc2(out1))

            out3 = self.Embedding(out2).tanh()
            norm = torch.norm(out3, p=2, dim=1, keepdim=True)
            out3 = out3 / norm
            return  out3  

class Data_Net(nn.Module):
    def __init__(self, input_dim=28*28, out_dim=20):
        super(Data_Net, self).__init__()
        mid_num1 = 4096
        mid_num2 = 4096
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)

        self.fc3 = nn.Linear(mid_num2, out_dim, bias=False)
        nn.init.uniform_( self.fc3.weight, -1. / np.sqrt(float(input_dim)), 1. / np.sqrt(float(input_dim)) )
    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))

        out3 = self.fc3(out2).tanh()
        norm = torch.norm(out3, p=2, dim=1, keepdim=True)
        out3 = out3 / norm
        return  out3
    
class Prototype_Net(nn.Module):
    def __init__(self, input_dim, num_class, num_prototypes_per_class, out_dim):
        super(Prototype_Net, self).__init__() 
        self.num_class = num_class
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_class * num_prototypes_per_class
        self.out_dim = out_dim
        self.p_emb = Prototype_embedding(num_class, out_dim).cuda()
        self.data_net = Data_Net(input_dim, out_dim)
    def get_pemb_weight(self):
        return copy.deepcopy(self.p_emb.state_dict())
    # def load_pemb_weight(self, weight_dict):
    #      self.p_emb.load_state_dict(weight_dict)
    def get_similaritys(self, x, prototype_vectors):
        cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6)
        sim = cos(x , prototype_vectors)
        return sim
    def forward(self, x, p_emb_inputs, prototype_vectors=None):
        data_fea = self.data_net(x)
        prototype_vectors = self.p_emb(p_emb_inputs) if prototype_vectors is None else prototype_vectors
        sim = self.get_similaritys(data_fea, prototype_vectors)
        return data_fea, prototype_vectors, sim

