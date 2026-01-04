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
    'HT1','Cool1_WQ','Cool1_AC','Cool1_FC','Cool1_FCAC',
    'HT2','Cool2_WQ','Cool2_AC','Cool2_FC','Cool2_FCAC']
PROC_SCALAR=['DeformTemp_C', 'DeformRate','HT1Temp_C','HT1Time_h','HT2Temp_C','HT2Time_h']
PROP=['YM', 'YS', 'UTS', 'El']
PROP_LABELS = {'YM':'YM(GPa)', 'YS':'YS(MPa)', 'UTS':'UTS(MPa)',
               'El':'El(%)', 'HV':'HV'}

N_ELEM = len(COMP)
N_ELEM_FEAT = 30
N_ELEM_FEAT_P1 = N_ELEM_FEAT + 1
N_PROC_BOOL = len(PROC_BOOL)
N_PROC_SCALAR = len(PROC_SCALAR)
N_PROP = len(PROP)
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
        self._num_fc_neuron = N_ELEM * N_ELEM_FEAT_P1 + N_PROC
        self.fc1 = nn.Linear(self._num_fc_neuron, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.ELU(0.2)

        self.reset_parameters()

        self.lr = LEARNING_RATE  # learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc):
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
        x = torch.cat([x, proc.reshape(-1, N_PROC)], dim=-1)
        
        # 全连接层
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNN_FCNN_MESH_Model(nn.Module):
    def __init__(self, temp=[1, 3, 64]):
        super(CNN_FCNN_MESH_Model, self).__init__()

        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size)
        self.bn1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size)
        self.bn2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        self._num_fc_neuron = N_ELEM * N_ELEM_FEAT_P1 + N_PROC_BOOL + N_PROC_SCALAR
        self._num_fc_comp_neuron = N_ELEM + N_PROC_BOOL + N_PROC_SCALAR
        self._num_neuron = temp[2]
        self._num_fc_layers = temp[1]
        self._num_cnn_layers = temp[0]

        self.fc1 = nn.Linear(self._num_fc_neuron, self._num_neuron)
        self.fc1_comp = nn.Linear(self._num_fc_comp_neuron, self._num_neuron)
        self.fc2 = nn.Linear(self._num_neuron, self._num_neuron)
        self.fc3 = nn.Linear(self._num_neuron, self._num_neuron)
        self.fc4 = nn.Linear(self._num_neuron, self._num_neuron)
        self.fc_prop = nn.Linear(self._num_neuron,  N_PROP)
        self.fc_0_prop = nn.Linear(self._num_fc_neuron,  N_PROP)
        self.fc_comp_prop = nn.Linear(self._num_fc_comp_neuron, N_PROP)

        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.ELU(0.2)

        self.reset_parameters()

        self.lr = LEARNING_RATE
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
        if self._num_cnn_layers==0:
            x = comp.reshape(-1, N_ELEM * 1)
            x = torch.cat([x, proc_bool.reshape(-1, N_PROC_BOOL), proc_scalar.reshape(-1, N_PROC_SCALAR)], dim=-1)
            if self._num_fc_layers==1:
                x=self.fc_comp_prop(x)
                return x
            elif self._num_fc_layers==2:
                x=self.leaky_relu(self.fc1_comp(x))
                x=self.dropout(x)
                x=self.fc_prop(x)
                return x
            elif self._num_fc_layers==3:
                x=self.leaky_relu(self.fc1_comp(x))
                x=self.dropout(x)
                x=self.leaky_relu(self.fc2(x))
                x=self.fc_prop(x)
                return x
        elif self._num_cnn_layers==1:
            x = torch.cat([comp, elem_feature], dim=-1)
            x = self.leaky_relu(self.bn1(self.conv1(x)))
            x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
            x = x.view(-1, N_ELEM * N_ELEM_FEAT_P1)
            x = torch.cat([x, proc_bool.reshape(-1, N_PROC_BOOL), proc_scalar.reshape(-1, N_PROC_SCALAR)], dim=-1)
        elif self._num_cnn_layers==2:
            x = torch.cat([comp, elem_feature], dim=-1)
            x = self.leaky_relu(self.bn1(self.conv1(x)))
            x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
            x = self.leaky_relu(self.bn2(self.conv2(x)))
            x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
            x = x.view(-1, N_ELEM * N_ELEM_FEAT_P1)
            x = torch.cat([x, proc_bool.reshape(-1, N_PROC_BOOL), proc_scalar.reshape(-1, N_PROC_SCALAR)], dim=-1)
        
        match self._num_fc_layers:
            case 1:
                x=self.fc_0_prop(x)
            case 2:
                x=self.leaky_relu(self.fc1(x))
                x=self.dropout(x)
                x=self.fc_prop(x)
            case 3:
                x=self.leaky_relu(self.fc1(x))
                x=self.dropout(x)
                x=self.leaky_relu(self.fc2(x))
                x=self.fc_prop(x)
            case 4:
                x=self.leaky_relu(self.fc1(x))
                x=self.dropout(x)
                x=self.leaky_relu(self.fc2(x))
                x=self.leaky_relu(self.fc3(x))
                x=self.fc_prop(x)

        result = x

        return result

class FCNN_Model(nn.Module):
    def __init__(self):
        super(FCNN_Model, self).__init__()

        self._n_in_fcnn = N_ELEM + N_ELEM_FEAT + N_PROC_BOOL + N_PROC_SCALAR
        self.fc1 = nn.Linear(self._n_in_fcnn, 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc_ym = nn.Linear(64, 32)
        self.fc_ys = nn.Linear(64, 32)
        self.fc_uts = nn.Linear(64, 32)
        self.fc_el = nn.Linear(64, 32)
        self.out_ym = nn.Linear(32, 1)
        self.out_ys = nn.Linear(32, 1)
        self.out_uts = nn.Linear(32, 1)
        self.out_el = nn.Linear(32, 1)

        self.af = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.reset_parameters()

        self.lr = LEARNING_RATE        
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
        mef = torch.sum(comp.squeeze(-1).squeeze(1).unsqueeze(-1) * elem_feature.squeeze(1), dim=1)
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            mef.reshape(-1, N_ELEM_FEAT), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR)], dim=-1)
        x = self.dropout(x)
        x = self.af(self.fc1(x))
        x = self.af(self.fc2(x))
        ym = self.out_ym(self.af(self.fc_ym(x)))
        ys = self.out_ys(self.af(self.fc_ys(x)))
        uts = self.out_uts(self.af(self.fc_uts(x)))
        el = self.out_el(self.af(self.fc_el(x)))
        x = torch.cat([ym, ys, uts, el], dim=-1)
        return x




if __name__ == '__main__':
    _batch_size = 8
    test_input = (torch.ones((_batch_size, 1, N_ELEM, 1)).to(device), \
                 torch.ones((_batch_size, 1, N_ELEM, N_ELEM_FEAT)).to(device), \
                 torch.ones((_batch_size, 1, N_PROC_BOOL, 1)).to(device), \
                 torch.ones((_batch_size, 1, N_PROC_SCALAR, 1)).to(device),)
    model = FCNN_Model().to(device)
    print(model(*test_input).size())