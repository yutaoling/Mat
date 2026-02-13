import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init

torch.autograd.set_detect_anomaly(True)
ID = ['id', 'Activated']
COMP = [
    'Ti', 'Al', 'V', 'Cr', 'Mn', 'Fe', 'Cu', 
    'Zr', 'Nb', 'Mo', 'Sn', 'Hf', 'Ta', 'W', 
    'Si', 'C', 'N', 'O', 'Sc'
    ]
PROC_BOOL=[
    'Is_Not_Wrought', 'Is_Wrought',
    'HT1_Quench','HT1_Air','HT1_Furnace',
    'HT2_Quench','HT2_Air','HT2_Furnace',
    ]
PROC_SCALAR=['Def_Temp', 'Def_Strain',
    'HT1_Temp','HT1_Time',
    'HT2_Temp','HT2_Time'
    ]
PHASE_SCALAR=['Mo_eq', 'Al_eq', 'beta_transform_T']
PROP=['YM', 'YS', 'UTS', 'El', 'HV']

N_YM=850
N_YS=935
N_UTS=945
N_EL=790
N_HV=185
N_PROP_SAMPLE = [N_YM, N_YS, N_UTS, N_EL, N_HV]
N_SUM_PROP_SAMPLE = sum(N_PROP_SAMPLE)


N_ELEM = len(COMP)
N_ELEM_FEAT = 30
N_ELEM_FEAT_P1 = N_ELEM_FEAT + 1
N_PROC_BOOL = len(PROC_BOOL)
N_PROC_SCALAR = len(PROC_SCALAR)
N_PHASE_SCALAR = len(PHASE_SCALAR)
N_PROP = len(PROP)

N_FC_NERON = 128
N_BRANCH_NERON = 64
LEARNING_RATE = 5e-4

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ELM_CPrPh(nn.Module):
    def __init__(self):
        super(ELM_CPrPh, self).__init__()
        
        self.fc1 = nn.Linear(N_ELEM + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR, N_FC_NERON)
        self.fc2 = nn.Linear(N_FC_NERON, 5)
        self.af = nn.LeakyReLU(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc1
        
        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp.reshape(-1, N_ELEM),
            proc_bool.reshape(-1, N_PROC_BOOL),
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x

class ELM_C(nn.Module):
    def __init__(self):
        super(ELM_C, self).__init__()
        
        self.fc1 = nn.Linear(N_ELEM, N_FC_NERON)
        self.fc2 = nn.Linear(N_FC_NERON, 5)
        self.af = nn.LeakyReLU(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc1

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = comp.reshape(-1, N_ELEM)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x

class ELM_CPh(nn.Module):
    def __init__(self):
        super(ELM_CPh, self).__init__()
        
        self.fc1 = nn.Linear(N_ELEM + N_PHASE_SCALAR, N_FC_NERON)
        self.fc2 = nn.Linear(N_FC_NERON, 5)
        self.af = nn.LeakyReLU(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc1

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp.reshape(-1, N_ELEM),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x

class ELM_CPr(nn.Module):
    def __init__(self):
        super(ELM_CPr, self).__init__()
        
        self.fc1 = nn.Linear(N_ELEM + N_PROC_BOOL + N_PROC_SCALAR, N_FC_NERON)
        self.fc2 = nn.Linear(N_FC_NERON, 5)
        self.af = nn.LeakyReLU(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc1

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp.reshape(-1, N_ELEM),
            proc_bool.reshape(-1, N_PROC_BOOL),
            proc_scalar.reshape(-1, N_PROC_SCALAR)], dim=-1)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x

class ELM_ElemFeat(nn.Module):
    def __init__(self):
        super(ELM_ElemFeat, self).__init__()
        
        self.fc1 = nn.Linear(N_ELEM + N_ELEM_FEAT + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR, N_FC_NERON)
        self.fc2 = nn.Linear(N_FC_NERON, 5)
        self.af = nn.LeakyReLU(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc1

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        mef = torch.sum(comp.squeeze(-1).squeeze(1).unsqueeze(-1) * elem_feature.squeeze(1), dim=1)
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            mef.reshape(-1, N_ELEM_FEAT), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x

class ELM_CNN(nn.Module):
    def __init__(self):
        super(ELM_CNN, self).__init__()

        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        self._n_cnn_out = N_ELEM * N_ELEM_FEAT_P1
        self._n_in_fcnn = self._n_cnn_out + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        
        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, 5)
        self.af = nn.LeakyReLU(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc1

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp, elem_feature], dim=-1)
        residual = x
        
        x = self.af(self.bn_conv1(self.conv1(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        x = self.af(self.bn_conv2(self.conv2(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        
        x = x + residual
        x = x.view(-1, self._n_cnn_out)
        x = torch.cat([
            x,
            proc_bool.reshape(-1, N_PROC_BOOL),
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)
        ], dim=-1)

        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()

        self._n_in_fcnn = N_ELEM + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON

        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)
        self.fc3 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn3 = nn.BatchNorm1d(self._n_fcnn)
        self.fc4 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn4 = nn.BatchNorm1d(self._n_fcnn)
        self.out = nn.Linear(self._n_fcnn, 5)

        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc4

        self.reset_parameters()

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        

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

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.af(self.bn2(self.fc2(x)))
        x = self.af(self.bn3(self.fc3(x)))
        x = self.af(self.bn4(self.fc4(x)))
        x = self.out(x)

        return x

class FCNN_MSHBranched(nn.Module):
    def __init__(self):
        super(FCNN_MSHBranched, self).__init__()

        self._n_in_fcnn = N_ELEM + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        self._n_branch = N_BRANCH_NERON

        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)

        self.fc_m1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_m1 = nn.BatchNorm1d(self._n_branch)
        self.fc_m2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_m = nn.Linear(self._n_branch, 1)

        self.fc_s1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_s1 = nn.BatchNorm1d(self._n_branch)
        self.fc_s2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_s = nn.Linear(self._n_branch, 3)

        self.fc_h1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_h1 = nn.BatchNorm1d(self._n_branch)
        self.fc_h2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_h = nn.Linear(self._n_branch, 1)

        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc2

        self.reset_parameters()

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        

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

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.af(self.bn2(self.fc2(x)))
        
        m = self.af(self.bn_m1(self.fc_m1(x)))
        m = self.af(self.fc_m2(m))
        m = self.out_m(m)

        s = self.af(self.bn_s1(self.fc_s1(x)))
        s = self.af(self.fc_s2(s))
        s = self.out_s(s)

        h = self.af(self.bn_h1(self.fc_h1(x)))
        h = self.af(self.fc_h2(h))
        h = self.out_h(h)
        
        x = torch.cat([m, s, h], dim=-1)
        return x

class FCNN_FullyBranched(nn.Module):
    def __init__(self):
        super(FCNN_FullyBranched, self).__init__()

        self._n_in_fcnn = N_ELEM + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        self._n_branch = N_BRANCH_NERON

        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)

        self.fc_ym1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_ym1 = nn.BatchNorm1d(self._n_branch)
        self.fc_ym2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ym = nn.Linear(self._n_branch, 1)

        self.fc_ys1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_ys1 = nn.BatchNorm1d(self._n_branch)
        self.fc_ys2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ys = nn.Linear(self._n_branch, 1)

        self.fc_uts1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_uts1 = nn.BatchNorm1d(self._n_branch)
        self.fc_uts2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_uts = nn.Linear(self._n_branch, 1)

        self.fc_el1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_el1 = nn.BatchNorm1d(self._n_branch)
        self.fc_el2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_el = nn.Linear(self._n_branch, 1)

        self.fc_hv1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_hv1 = nn.BatchNorm1d(self._n_branch)
        self.fc_hv2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_hv = nn.Linear(self._n_branch, 1)

        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc2

        self.reset_parameters()

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        

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

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.af(self.bn2(self.fc2(x)))
        
        ym = self.af(self.bn_ym1(self.fc_ym1(x)))
        ym = self.af(self.fc_ym2(ym))
        ym = self.out_ym(ym)

        ys = self.af(self.bn_ys1(self.fc_ys1(x)))
        ys = self.af(self.fc_ys2(ys))
        ys = self.out_ys(ys)

        uts = self.af(self.bn_uts1(self.fc_uts1(x)))
        uts = self.af(self.fc_uts2(uts))
        uts = self.out_uts(uts)

        el = self.af(self.bn_el1(self.fc_el1(x)))
        el = self.af(self.fc_el2(el))
        el = self.out_el(el)

        hv = self.af(self.bn_hv1(self.fc_hv1(x)))
        hv = self.af(self.fc_hv2(hv))
        hv = self.out_hv(hv)
        
        x = torch.cat([ym, ys, uts, el, hv], dim=-1)
        return x

class FCNN_ElemFeat(nn.Module):
    def __init__(self):
        super(FCNN_ElemFeat, self).__init__()

        self._n_in_fcnn = N_ELEM + N_ELEM_FEAT + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON

        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)
        self.fc3 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn3 = nn.BatchNorm1d(self._n_fcnn)
        self.fc4 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn4 = nn.BatchNorm1d(self._n_fcnn)
        self.out = nn.Linear(self._n_fcnn, 5)

        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc4

        self.reset_parameters()

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        

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

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        mef = torch.sum(comp.squeeze(-1).squeeze(1).unsqueeze(-1) * elem_feature.squeeze(1), dim=1)
        
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            mef.reshape(-1, N_ELEM_FEAT), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.af(self.bn2(self.fc2(x)))
        x = self.af(self.bn3(self.fc3(x)))
        x = self.af(self.bn4(self.fc4(x)))
        x = self.out(x)

        return x

class FCNN_ElemFeat_MSHBranched(nn.Module):
    def __init__(self):
        super(FCNN_ElemFeat_MSHBranched, self).__init__()

        self._n_in_fcnn = N_ELEM + N_ELEM_FEAT + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        self._n_branch = N_BRANCH_NERON

        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)

        self.fc_m1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_m1 = nn.BatchNorm1d(self._n_branch)
        self.fc_m2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_m = nn.Linear(self._n_branch, 1)

        self.fc_s1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_s1 = nn.BatchNorm1d(self._n_branch)
        self.fc_s2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_s = nn.Linear(self._n_branch, 3)

        self.fc_h1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_h1 = nn.BatchNorm1d(self._n_branch)
        self.fc_h2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_h = nn.Linear(self._n_branch, 1)

        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc2

        self.reset_parameters()

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        

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

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        mef = torch.sum(comp.squeeze(-1).squeeze(1).unsqueeze(-1) * elem_feature.squeeze(1), dim=1)
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            mef.reshape(-1, N_ELEM_FEAT), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.af(self.bn2(self.fc2(x)))
        
        m = self.af(self.bn_m1(self.fc_m1(x)))
        m = self.af(self.fc_m2(m))
        m = self.out_m(m)

        s = self.af(self.bn_s1(self.fc_s1(x)))
        s = self.af(self.fc_s2(s))
        s = self.out_s(s)

        h = self.af(self.bn_h1(self.fc_h1(x)))
        h = self.af(self.fc_h2(h))
        h = self.out_h(h)
        
        x = torch.cat([m, s, h], dim=-1)
        return x

class FCNN_ElemFeat_FullyBranched(nn.Module):
    def __init__(self):
        super(FCNN_ElemFeat_FullyBranched, self).__init__()

        self._n_in_fcnn = N_ELEM + N_ELEM_FEAT + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        self._n_branch = N_BRANCH_NERON
        
        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)

        self.fc_ym1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_ym1 = nn.BatchNorm1d(self._n_branch)
        self.fc_ym2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ym = nn.Linear(self._n_branch, 1)

        self.fc_ys1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_ys1 = nn.BatchNorm1d(self._n_branch)
        self.fc_ys2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ys = nn.Linear(self._n_branch, 1)

        self.fc_uts1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_uts1 = nn.BatchNorm1d(self._n_branch)
        self.fc_uts2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_uts = nn.Linear(self._n_branch, 1)

        self.fc_el1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_el1 = nn.BatchNorm1d(self._n_branch)
        self.fc_el2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_el = nn.Linear(self._n_branch, 1)

        self.fc_hv1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_hv1 = nn.BatchNorm1d(self._n_branch)
        self.fc_hv2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_hv = nn.Linear(self._n_branch, 1)

        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc2

        self.reset_parameters()

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        

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

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        mef = torch.sum(comp.squeeze(-1).squeeze(1).unsqueeze(-1) * elem_feature.squeeze(1), dim=1)
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            mef.reshape(-1, N_ELEM_FEAT), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.af(self.bn2(self.fc2(x)))

        ym = self.af(self.bn_ym1(self.fc_ym1(x)))
        ym = self.af(self.fc_ym2(ym))
        ym = self.out_ym(ym)

        ys = self.af(self.bn_ys1(self.fc_ys1(x)))
        ys = self.af(self.fc_ys2(ys))
        ys = self.out_ys(ys)

        uts = self.af(self.bn_uts1(self.fc_uts1(x)))
        uts = self.af(self.fc_uts2(uts))
        uts = self.out_uts(uts)

        el = self.af(self.bn_el1(self.fc_el1(x)))
        el = self.af(self.fc_el2(el))
        el = self.out_el(el)

        hv = self.af(self.bn_hv1(self.fc_hv1(x)))
        hv = self.af(self.fc_hv2(hv))
        hv = self.out_hv(hv)
        
        x = torch.cat([ym, ys, uts, el, hv], dim=-1)
        return x

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        self._n_cnn_out = N_ELEM * N_ELEM_FEAT_P1
        self._n_in_fcnn = self._n_cnn_out + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        
        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)
        self.out = nn.Linear(self._n_fcnn, 5)
        
        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc2

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1.)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp, elem_feature], dim=-1)
        residual = x
        
        x = self.af(self.bn_conv1(self.conv1(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        x = self.af(self.bn_conv2(self.conv2(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        
        x = x + residual
        x = x.view(-1, self._n_cnn_out)
        x = torch.cat([
            x,
            proc_bool.reshape(-1, N_PROC_BOOL),
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)
        ], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.af(self.bn2(self.fc2(x)))
        x = self.out(x)
        
        return x

class CNN_MSHBranched(nn.Module):

    def __init__(self):
        super(CNN_MSHBranched, self).__init__()
        
        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        self._n_cnn_out = N_ELEM * N_ELEM_FEAT_P1
        self._n_in_fcnn = self._n_cnn_out + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        self._n_branch = N_BRANCH_NERON
        
        self.fc_m1 = nn.Linear(self._n_in_fcnn, self._n_branch)
        self.bn_m1 = nn.BatchNorm1d(self._n_branch)
        self.fc_m2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_m = nn.Linear(self._n_branch, 1)
        
        self.fc_s1 = nn.Linear(self._n_in_fcnn, self._n_branch)
        self.bn_s1 = nn.BatchNorm1d(self._n_branch)
        self.fc_s2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_s = nn.Linear(self._n_branch, 3)
        
        self.fc_v1 = nn.Linear(self._n_in_fcnn, self._n_branch)
        self.bn_v1 = nn.BatchNorm1d(self._n_branch)
        self.fc_v2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_v = nn.Linear(self._n_branch, 1)
        
        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.conv2

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1.)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp, elem_feature], dim=-1)
        residual = x
        
        x = self.af(self.bn_conv1(self.conv1(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        x = self.af(self.bn_conv2(self.conv2(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        
        x = x + residual
        x = x.view(-1, self._n_cnn_out)
        x = torch.cat([
            x,
            proc_bool.reshape(-1, N_PROC_BOOL),
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)
        ], dim=-1)
        
        m = self.af(self.bn_m1(self.fc_m1(x)))
        m = self.af(self.fc_m2(m))
        m = self.out_m(m)
        
        s = self.af(self.bn_s1(self.fc_s1(x)))
        s = self.af(self.fc_s2(s))
        s = self.out_s(s)
        
        v = self.af(self.bn_v1(self.fc_v1(x)))
        v = self.af(self.fc_v2(v))
        v = self.out_v(v)
        
        out = torch.cat([m, s, v], dim=-1)
        return out

class CNN_FullyBranched(nn.Module):

    def __init__(self):
        super(CNN_FullyBranched, self).__init__()
        
        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        self._n_cnn_out = N_ELEM * N_ELEM_FEAT_P1
        self._n_in_fcnn = self._n_cnn_out + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        self._n_branch = N_BRANCH_NERON
        
        self.fc_ym1 = nn.Linear(self._n_in_fcnn, self._n_branch)
        self.bn_ym1 = nn.BatchNorm1d(self._n_branch)
        self.fc_ym2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ym = nn.Linear(self._n_branch, 1)

        self.fc_ys1 = nn.Linear(self._n_in_fcnn, self._n_branch)
        self.bn_ys1 = nn.BatchNorm1d(self._n_branch)
        self.fc_ys2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ys = nn.Linear(self._n_branch, 1)

        self.fc_uts1 = nn.Linear(self._n_in_fcnn, self._n_branch)
        self.bn_uts1 = nn.BatchNorm1d(self._n_branch)
        self.fc_uts2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_uts = nn.Linear(self._n_branch, 1)

        self.fc_el1 = nn.Linear(self._n_in_fcnn, self._n_branch)
        self.bn_el1 = nn.BatchNorm1d(self._n_branch)
        self.fc_el2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_el = nn.Linear(self._n_branch, 1)

        self.fc_hv1 = nn.Linear(self._n_in_fcnn, self._n_branch)
        self.bn_hv1 = nn.BatchNorm1d(self._n_branch)
        self.fc_hv2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_hv = nn.Linear(self._n_branch, 1)
        
        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.conv2

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1.)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp, elem_feature], dim=-1)
        residual = x
        
        x = self.af(self.bn_conv1(self.conv1(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        x = self.af(self.bn_conv2(self.conv2(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        
        x = x + residual
        
        x = x.view(-1, self._n_cnn_out)
        
        x = torch.cat([
            x,
            proc_bool.reshape(-1, N_PROC_BOOL),
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)
        ], dim=-1)

        ym = self.af(self.bn_ym1(self.fc_ym1(x)))
        ym = self.af(self.fc_ym2(ym))
        ym = self.out_ym(ym)

        ys = self.af(self.bn_ys1(self.fc_ys1(x)))
        ys = self.af(self.fc_ys2(ys))
        ys = self.out_ys(ys)

        uts = self.af(self.bn_uts1(self.fc_uts1(x)))
        uts = self.af(self.fc_uts2(uts))
        uts = self.out_uts(uts)

        el = self.af(self.bn_el1(self.fc_el1(x)))
        el = self.af(self.fc_el2(el))
        el = self.out_el(el)

        hv = self.af(self.bn_hv1(self.fc_hv1(x)))
        hv = self.af(self.fc_hv2(hv))
        hv = self.out_hv(hv)
        
        x = torch.cat([ym, ys, uts, el, hv], dim=-1)
        return x

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.base_feat_dim = 32
        self.base_encoder = nn.Sequential(
            nn.Linear(N_ELEM_FEAT, self.base_feat_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.attn_hidden_dim = 64
        
        self.elem_encoder = nn.Sequential(
            nn.Linear(N_ELEM_FEAT, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.attn_hidden_dim, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attn_hidden_dim, 
            num_heads=self.num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        self.comp_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self.attn_proj = nn.Sequential(
            nn.Linear(self.attn_hidden_dim, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.proc_phase_dim = N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self.fusion_input_dim = self.base_feat_dim + self.attn_hidden_dim + self.proc_phase_dim
        self.global_hidden_dim = N_FC_NERON

        self.fc_main = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.branch_dim = N_BRANCH_NERON

        self.head_x = self._make_head(self.branch_dim, 5)

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.head_x[0]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_head(self, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(self.global_hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        batch_size = comp.size(0)
        
        comp_sq = comp.squeeze(-1).squeeze(1)
        feat_sq = elem_feature.squeeze(1)
        
        base_feat = self.base_encoder(feat_sq)
        comp_weights = comp_sq.unsqueeze(-1)
        base_emb = torch.sum(comp_weights * base_feat, dim=1)
        
        elem_emb = self.elem_encoder(feat_sq)
        
        gate = self.comp_gate(comp_sq.unsqueeze(-1))
        elem_emb_gated = elem_emb * gate
        
        attn_out, attn_weights = self.attention(
            elem_emb_gated, elem_emb_gated, elem_emb_gated
        )
        
        attn_out = self.attn_proj(attn_out)
        attn_emb = torch.sum(comp_weights * attn_out, dim=1)
        
        proc_phase = torch.cat([
            proc_bool.reshape(batch_size, -1), 
            proc_scalar.reshape(batch_size, -1),
            phase_scalar.reshape(batch_size, -1)
        ], dim=-1)
        
        x = torch.cat([base_emb, attn_emb, proc_phase], dim=-1)
        x = self.fc_main(x)

        x = self.head_x(x)

        return x
    
    def get_attention_weights(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        self.eval()
        with torch.no_grad():
            comp_sq = comp.squeeze(-1).squeeze(1)
            feat_sq = elem_feature.squeeze(1)
            
            elem_emb = self.elem_encoder(feat_sq)
            gate = self.comp_gate(comp_sq.unsqueeze(-1))
            elem_emb_gated = elem_emb * gate
            
            _, attn_weights = self.attention(
                elem_emb_gated, elem_emb_gated, elem_emb_gated
            )
            
        return attn_weights, gate.squeeze(-1)

class Attention_MSHBranched(nn.Module):
    def __init__(self):
        super(Attention_MSHBranched, self).__init__()

        self.base_feat_dim = 32
        self.base_encoder = nn.Sequential(
            nn.Linear(N_ELEM_FEAT, self.base_feat_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.attn_hidden_dim = 64
        
        self.elem_encoder = nn.Sequential(
            nn.Linear(N_ELEM_FEAT, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.attn_hidden_dim, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attn_hidden_dim, 
            num_heads=self.num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        self.comp_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self.attn_proj = nn.Sequential(
            nn.Linear(self.attn_hidden_dim, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.proc_phase_dim = N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self.fusion_input_dim = self.base_feat_dim + self.attn_hidden_dim + self.proc_phase_dim
        self.global_hidden_dim = N_FC_NERON

        self.fc_main = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.branch_dim = N_BRANCH_NERON

        self.head_ym = self._make_head(self.branch_dim, 1)
        self.head_s = self._make_head(self.branch_dim, 3)
        self.head_hv = self._make_head(self.branch_dim, 1)

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc_main[4]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_head(self, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(self.global_hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        batch_size = comp.size(0)
        
        comp_sq = comp.squeeze(-1).squeeze(1)
        feat_sq = elem_feature.squeeze(1)
        
        base_feat = self.base_encoder(feat_sq)
        comp_weights = comp_sq.unsqueeze(-1)
        base_emb = torch.sum(comp_weights * base_feat, dim=1)
        
        elem_emb = self.elem_encoder(feat_sq)
        
        gate = self.comp_gate(comp_sq.unsqueeze(-1))
        elem_emb_gated = elem_emb * gate
        
        attn_out, attn_weights = self.attention(
            elem_emb_gated, elem_emb_gated, elem_emb_gated
        )
        
        attn_out = self.attn_proj(attn_out)
        attn_emb = torch.sum(comp_weights * attn_out, dim=1)
        
        proc_phase = torch.cat([
            proc_bool.reshape(batch_size, -1), 
            proc_scalar.reshape(batch_size, -1),
            phase_scalar.reshape(batch_size, -1)
        ], dim=-1)
        
        x = torch.cat([base_emb, attn_emb, proc_phase], dim=-1)
        x = self.fc_main(x)

        ym = self.head_ym(x)
        s = self.head_s(x)
        hv = self.head_hv(x)

        out = torch.cat([ym, s, hv], dim=-1)

        return out
    
    def get_attention_weights(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        self.eval()
        with torch.no_grad():
            comp_sq = comp.squeeze(-1).squeeze(1)
            feat_sq = elem_feature.squeeze(1)
            
            elem_emb = self.elem_encoder(feat_sq)
            gate = self.comp_gate(comp_sq.unsqueeze(-1))
            elem_emb_gated = elem_emb * gate
            
            _, attn_weights = self.attention(
                elem_emb_gated, elem_emb_gated, elem_emb_gated
            )
            
        return attn_weights, gate.squeeze(-1)

class Attention_FullyBranched(nn.Module):
    def __init__(self):
        super(Attention_FullyBranched, self).__init__()

        self.base_feat_dim = 32
        self.base_encoder = nn.Sequential(
            nn.Linear(N_ELEM_FEAT, self.base_feat_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.attn_hidden_dim = 64
        
        self.elem_encoder = nn.Sequential(
            nn.Linear(N_ELEM_FEAT, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.attn_hidden_dim, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attn_hidden_dim, 
            num_heads=self.num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        self.comp_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self.attn_proj = nn.Sequential(
            nn.Linear(self.attn_hidden_dim, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.proc_phase_dim = N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self.fusion_input_dim = self.base_feat_dim + self.attn_hidden_dim + self.proc_phase_dim
        self.global_hidden_dim = N_FC_NERON

        self.fc_main = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.branch_dim = N_BRANCH_NERON

        self.head_ym = self._make_head(self.branch_dim, 1)
        self.head_ys = self._make_head(self.branch_dim, 1)
        self.head_uts = self._make_head(self.branch_dim, 1)
        self.head_el = self._make_head(self.branch_dim, 1)
        self.head_hv = self._make_head(self.branch_dim, 1)

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc_main[4]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_head(self, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(self.global_hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        batch_size = comp.size(0)
        
        comp_sq = comp.squeeze(-1).squeeze(1)
        feat_sq = elem_feature.squeeze(1)
        
        base_feat = self.base_encoder(feat_sq)
        comp_weights = comp_sq.unsqueeze(-1)
        base_emb = torch.sum(comp_weights * base_feat, dim=1)
        
        elem_emb = self.elem_encoder(feat_sq)
        
        gate = self.comp_gate(comp_sq.unsqueeze(-1))
        elem_emb_gated = elem_emb * gate
        
        attn_out, attn_weights = self.attention(
            elem_emb_gated, elem_emb_gated, elem_emb_gated
        )
        
        attn_out = self.attn_proj(attn_out)
        attn_emb = torch.sum(comp_weights * attn_out, dim=1)
        
        proc_phase = torch.cat([
            proc_bool.reshape(batch_size, -1), 
            proc_scalar.reshape(batch_size, -1),
            phase_scalar.reshape(batch_size, -1)
        ], dim=-1)
        
        x = torch.cat([base_emb, attn_emb, proc_phase], dim=-1)
        x = self.fc_main(x)

        ym = self.head_ym(x)
        ys = self.head_ys(x)
        uts = self.head_uts(x)
        el = self.head_el(x)
        hv = self.head_hv(x)

        out = torch.cat([ym, ys, uts, el, hv], dim=-1)

        return out
    
    def get_attention_weights(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        self.eval()
        with torch.no_grad():
            comp_sq = comp.squeeze(-1).squeeze(1)
            feat_sq = elem_feature.squeeze(1)
            
            elem_emb = self.elem_encoder(feat_sq)
            gate = self.comp_gate(comp_sq.unsqueeze(-1))
            elem_emb_gated = elem_emb * gate
            
            _, attn_weights = self.attention(
                elem_emb_gated, elem_emb_gated, elem_emb_gated
            )
            
        return attn_weights, gate.squeeze(-1)

class TiAlloyNet(nn.Module):
    def __init__(self):
        super(TiAlloyNet, self).__init__()

        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        self._n_cnn_out = N_ELEM * N_ELEM_FEAT_P1
        self._n_in_fcnn = self._n_cnn_out + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        self._n_branch = N_BRANCH_NERON
        
        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.out_1 = nn.Linear(self._n_fcnn, 3)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)

        self.fc_uts1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_uts1 = nn.BatchNorm1d(self._n_branch)
        self.fc_uts2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_uts = nn.Linear(self._n_branch, 1)

        self.fc_el1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_el1 = nn.BatchNorm1d(self._n_branch)
        self.fc_el2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_el = nn.Linear(self._n_branch, 1)

        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

        self.loss_weights = nn.Parameter(torch.ones(5, device=device))
        self.shared_layer = self.fc2
        self.reset_parameters()

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        

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

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp, elem_feature], dim=-1)
        residual = x
        
        x = self.af(self.bn_conv1(self.conv1(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        x = self.af(self.bn_conv2(self.conv2(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        
        x = x + residual
        
        x = x.view(-1, self._n_cnn_out)

        x = torch.cat([
            x,
            proc_bool.reshape(-1, N_PROC_BOOL),
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)
        ], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        out_1 = self.out_1(x)
        x = self.dropout(x)
        x = self.af(self.bn2(self.fc2(x)))

        uts = self.af(self.bn_uts1(self.fc_uts1(x)))
        uts = self.af(self.fc_uts2(uts))
        uts = self.out_uts(uts)

        el = self.af(self.bn_el1(self.fc_el1(x)))
        el = self.af(self.fc_el2(el))
        el = self.out_el(el)
        
        x = torch.cat([out_1[:,0:1], out_1[:,1:2], uts, el, out_1[:,2:3]], dim=-1)
        return x


if __name__ == '__main__':
    _batch_size = 8
    
    test_input = (
        torch.ones((_batch_size, 1, N_ELEM, 1)).to(device),
        torch.ones((_batch_size, 1, N_ELEM, N_ELEM_FEAT)).to(device),
        torch.ones((_batch_size, 1, N_PROC_BOOL, 1)).to(device),
        torch.ones((_batch_size, 1, N_PROC_SCALAR, 1)).to(device),
        torch.ones((_batch_size, 1, N_PHASE_SCALAR, 1)).to(device)
    )
    
    model_list = [
        ELM_CPrPh().to(device), ELM_C().to(device), ELM_CPh().to(device), ELM_CPr().to(device), ELM_ElemFeat().to(device), ELM_CNN().to(device),
        FCNN().to(device), FCNN_MSHBranched().to(device), FCNN_FullyBranched().to(device),
        FCNN_ElemFeat().to(device), FCNN_ElemFeat_MSHBranched().to(device), FCNN_ElemFeat_FullyBranched().to(device),
        Attention().to(device), Attention_MSHBranched().to(device), Attention_FullyBranched().to(device),
        TiAlloyNet().to(device)
    ]
    for model in model_list:
        print(f"{model(*test_input).size()}")
    