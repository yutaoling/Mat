'''Ref_2023_npj Comp Mat_A neural network model for high entropy alloy design'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init

torch.autograd.set_detect_anomaly(True)

# number of elements
N_ELEM = 10
# number of elemental fetures
N_ELEM_FEAT = 30
# number of elemental fetures + 1
N_ELEM_FEAT_P1 = N_ELEM_FEAT + 1
# number of process conditions
N_PROC_BOOL = 15
N_PROC_SCALAR = 6

N_PROP = 5
# learning rate
LEARNING_RATE = 5e-4

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        # interpret self.shape as the shape *excluding* the batch dimension
        return x.view(x.size(0), *self.shape)


class CnnDnnModel(nn.Module):
    '''
        CNN, ELU, batch normalization
        DNN, ELU, drop out
        Attention mech.
        Residual.

        # nn.Conv2d default: stride = 1, padding = 0
    '''
    def __init__(self):
        super(CnnDnnModel, self).__init__()

        # 卷积层
        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self._num_fc0_neuron = N_ELEM * N_ELEM_FEAT_P1 + 4
        self.CompCNN = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size),
            nn.BatchNorm2d(N_ELEM_FEAT_P1),
            nn.ReLU(),
            Reshape(1, N_ELEM, N_ELEM_FEAT_P1),

            nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size),
            nn.BatchNorm2d(N_ELEM_FEAT_P1),
            nn.ReLU(),
            Reshape(1, N_ELEM, N_ELEM_FEAT_P1),
        )
        # 全连接层
        self.FC0 = nn.Sequential(
            nn.Linear(self._num_fc0_neuron, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.Deform=nn.Sequential(
            nn.Linear(130,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
        )

        self.HeatTreatment=nn.Sequential(
            nn.Linear(134,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
        )

        self.fc11 = nn.Linear(128, 64)
        self.fc12 = nn.Linear(64, 1)
        self.fc21 = nn.Linear(128, 64)
        self.fc22 = nn.Linear(64, 1)
        self.fc31 = nn.Linear(128, 64)
        self.fc32 = nn.Linear(64, 1)
        self.fc41 = nn.Linear(128, 64)
        self.fc42 = nn.Linear(64, 1)
        self.fc51 = nn.Linear(128, 64)
        self.fc52 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.ReLU()

        self.reset_parameters()

        self.lr = LEARNING_RATE  # learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None: m.weight.data.fill_(1.)
                if m.bias is not None: m.bias.data.zero_()


    def forward(self, comp, elem_feature, proc_bool, proc_scalar):
        '''
            comp: (batch_size, 1, number_of_elements, 1)
            elem_feature: (batch_size, 1, number_of_elements, number_of_elemental_features), NOTE alway fixed
            prop: (batch_size, 1, N_PROC, 1)

            NOTE:
                For the model, elem_feature seems to be different for each sample,
                but ...
        '''
        proc_bool_flat = proc_bool.view(comp.size(0), -1)
        proc_scalar_flat = proc_scalar.view(comp.size(0), -1)


        x = torch.cat([comp, elem_feature], dim=-1)
        
        residual = x

        x = self.CompCNN(x)

        # 残差连接（避免就地修改，保留用于反向传播的中间张量）
        x = x + residual

        # 展平
        x = x.view(-1, N_ELEM * N_ELEM_FEAT_P1)
        # 链接initial condition
        x = torch.cat([x, proc_bool_flat[:,0:4]], dim=1)

        # 全连接层
        x = self.FC0(x)

        mask_deform = proc_bool_flat[:, 4].bool()
        if mask_deform.any():
            idx = mask_deform.nonzero(as_tuple=True)[0]
            x_sub = x[idx]
            add_feat = proc_scalar_flat[idx, 0:2]
            x_sub = torch.cat([x_sub, add_feat], dim=1)
            x_sub = self.Deform(x_sub)
            x_new = x.clone()
            x_new[idx] = x_sub
            x = x_new

        mask_ht1 = proc_bool_flat[:, 5].bool()
        if mask_ht1.any():
            idx = mask_ht1.nonzero(as_tuple=True)[0]
            x_sub = x[idx]
            add_bool = proc_bool_flat[idx, 6:10]      # (n_sub,4)
            add_scalar = proc_scalar_flat[idx, 2:4]   # (n_sub,2)
            add_feat = torch.cat([add_bool, add_scalar], dim=1)  # (n_sub,6)
            x_sub = torch.cat([x_sub, add_feat], dim=1)
            x_sub = self.HeatTreatment(x_sub)
            x_new = x.clone()
            x_new[idx] = x_sub
            x = x_new

        mask_ht2 = proc_bool_flat[:, 10].bool()
        if mask_ht2.any():
            idx = mask_ht2.nonzero(as_tuple=True)[0]
            x_sub = x[idx]
            add_bool = proc_bool_flat[idx, 11:15]
            add_scalar = proc_scalar_flat[idx, 4:6]
            add_feat = torch.cat([add_bool, add_scalar], dim=1)
            x_sub = torch.cat([x_sub, add_feat], dim=1)
            x_sub = self.HeatTreatment(x_sub)
            x_new = x.clone()
            x_new[idx] = x_sub
            x = x_new

        x1 = self.leaky_relu(self.fc11(x))
        x1 = self.fc12(x1)
        x2 = self.leaky_relu(self.fc21(x))
        x2 = self.fc22(x2)
        x3 = self.leaky_relu(self.fc31(x))
        x3 = self.fc32(x3)
        x4 = self.leaky_relu(self.fc41(x))
        x4 = self.fc42(x4)
        x5 = self.leaky_relu(self.fc51(x))
        x5 = self.fc52(x5)
        result = torch.cat((x1, x2, x3, x4, x5), dim=1)


        return result

if __name__ == '__main__':
    _batch_size = 8
    test_input = (torch.ones((_batch_size, 1, N_ELEM, 1)).to(device), \
                 torch.ones((_batch_size, 1, N_ELEM, N_ELEM_FEAT)).to(device), \
                 torch.ones((_batch_size, 1, N_PROC_BOOL, 1)).to(device), \
                 torch.ones((_batch_size, 1, N_PROC_SCALAR, 1)).to(device))
    model = CnnDnnModel().to(device)
    print(model(*test_input).size())