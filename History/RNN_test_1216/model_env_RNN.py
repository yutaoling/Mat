'''Ref_2023_npj Comp Mat_A neural network model for high entropy alloy design'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init

torch.autograd.set_detect_anomaly(True)

COMP = ['Ti', 'Al', 'Zr', 'Mo', 'V', 'Ta', 'Nb', 'Cr', 'Fe', 'Sn']

PROC_BOOL=[
    'InitStat_Case', 'InitStat_Rolled', 'InitStat_Forged', 'InitStat_Extrusion',
    'Deform',
    'HT1','Cool1_WQ','Cool1_AQ','Cool1_FC','Cool1_FCAC',
    'HT2','Cool2_WQ','Cool2_AC','Cool2_FC','Cool2_FCAC']
PROC_SCALAR=['DeformTemp_C', 'DeformRate','HT1Temp_C','HT1Time_h','HT2Temp_C','HT2Time_h']
PROP=['YM', 'YS', 'UTS', 'El', 'HV']
PROP_LABELS = {'YM':'YM(GPa)', 'YS':'YS(MPa)', 'UTS':'UTS(MPa)',
               'El':'El(%)', 'HV':'HV'}

# number of elements
N_ELEM = len(COMP)
# number of elemental fetures
N_ELEM_FEAT = 30
# number of elemental fetures + 1
N_ELEM_FEAT_P1 = N_ELEM_FEAT + 1
# number of process conditions
N_PROC_BOOL = len(PROC_BOOL)
N_PROC_SCALAR = len(PROC_SCALAR)

N_PROP = len(PROP)
# learning rate
LEARNING_RATE = 5e-4

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.Linear(self._num_fc0_neuron, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.Deform=nn.Sequential(
            nn.Linear(64+2,64),
            nn.ReLU(),
        )

        self.HeatTreatment=nn.Sequential(
            nn.Linear(64+6,64),
            nn.ReLU(),
        )

        self.Prop=nn.Sequential(
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16, N_PROP),
        )

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

        x = self.Prop(x)

        return x

if __name__ == '__main__':
    _batch_size = 8
    test_input = (torch.ones((_batch_size, 1, N_ELEM, 1)).to(device), \
                 torch.ones((_batch_size, 1, N_ELEM, N_ELEM_FEAT)).to(device), \
                 torch.ones((_batch_size, 1, N_PROC_BOOL, 1)).to(device), \
                 torch.ones((_batch_size, 1, N_PROC_SCALAR, 1)).to(device))
    model = CnnDnnModel().to(device)
    print(model(*test_input).size())