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
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size)
        self.bn1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size)
        self.bn2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        # self.attention = SelfAttention(1)

        # 全连接层
        self._num_fc_neuron = N_ELEM * N_ELEM_FEAT_P1 + N_PROC_BOOL + N_PROC_SCALAR
        self.fc1 = nn.Linear(self._num_fc_neuron, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc4 = nn.Linear(128, N_PROP)


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
        self.leaky_relu = nn.ELU(0.2)

        self.reset_parameters()

        self.lr = LEARNING_RATE  # learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        # self.fc4.weight.data.uniform_(*hidden_init(self.fc4))

        self.fc11.weight.data.uniform_(*hidden_init(self.fc11))
        self.fc12.weight.data.uniform_(*hidden_init(self.fc12))
        self.fc21.weight.data.uniform_(*hidden_init(self.fc21))
        self.fc22.weight.data.uniform_(*hidden_init(self.fc22))
        self.fc31.weight.data.uniform_(*hidden_init(self.fc31))
        self.fc32.weight.data.uniform_(*hidden_init(self.fc32))
        self.fc41.weight.data.uniform_(*hidden_init(self.fc41))
        self.fc42.weight.data.uniform_(*hidden_init(self.fc42))
        self.fc51.weight.data.uniform_(*hidden_init(self.fc51))
        self.fc52.weight.data.uniform_(*hidden_init(self.fc52))


    def forward(self, comp, elem_feature, proc_bool, proc_scalar):
        '''
            comp: (batch_size, 1, number_of_elements, 1)
            elem_feature: (batch_size, 1, number_of_elements, number_of_elemental_features), NOTE alway fixed
            prop: (batch_size, 1, N_PROC, 1)

            NOTE:
                For the model, elem_feature seems to be different for each sample,
                but ...
        '''
        x = torch.cat([comp, elem_feature], dim=-1)
        residual = x

        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)

        # 残差连接
        x += residual

        # 展平
        x = x.view(-1, N_ELEM * N_ELEM_FEAT_P1)

        # 链接process condition
        x = torch.cat([x, proc_bool.reshape(-1, N_PROC_BOOL), proc_scalar.reshape(-1, N_PROC_SCALAR)], dim=-1)

        # 全连接层
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        '''
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        '''

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
                 torch.ones((_batch_size, 1, N_PROC, 1)).to(device))
    model = CnnDnnModel().to(device)
    print(model(*test_input).size())