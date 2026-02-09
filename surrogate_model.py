import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

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
LEAKY_RATE = 0.2
DROP_RATE = 0.2

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class MaskedInputLayer(nn.Module):
    def __init__(self, use_random_sample=False):
        super(MaskedInputLayer, self).__init__()
        self.use_random_sample = use_random_sample

    def forward(self, x, mask, group_mean=None):
        if mask is None:
            return x
        valid_part = x * mask
        missing_mask = 1 - mask
        with torch.no_grad():
            if self.use_random_sample: # use random sample
                for feat_idx in range(x.shape[1]):
                    valid_vals = x[:, feat_idx][mask[:, feat_idx] > 0]
                    if valid_vals.numel() > 0:
                        sampled = valid_vals[torch.randint(0, len(valid_vals), (x.shape[0],))]
                        x[:, feat_idx] += missing_mask[:, feat_idx] * sampled
                    else:
                        x[:, feat_idx] += missing_mask[:, feat_idx] * group_mean[feat_idx]
            else: # use group mean
                replace_val = group_mean.expand_as(x)
                x = valid_part + missing_mask * replace_val
        return x

class ELM(nn.Module):
    def __init__(self, mask_mode = 'zero', Pr = True, Ph = True):
        super(ELM, self).__init__()
        self.mask_mode = mask_mode
        self.Pr = Pr
        self.Ph = Ph
        self._n_in_fcnn = N_ELEM
        if self.Pr:
            self._n_in_fcnn += N_PROC_BOOL + N_PROC_SCALAR
        if self.Ph:
            self._n_in_fcnn += N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON

        self.fc1 = nn.Linear(self._n_in_fcnn, N_FC_NERON)
        self.fc2 = nn.Linear(N_FC_NERON, N_PROP)
        self.af = nn.LeakyReLU(LEAKY_RATE)
        
        if self.mask_mode == 'learned':
            self.missing_proc_bool = nn.Parameter(torch.zeros(N_PROC_BOOL))
            self.missing_proc_scalar = nn.Parameter(torch.zeros(N_PROC_SCALAR))
            self.missing_phase_scalar = nn.Parameter(torch.zeros(N_PHASE_SCALAR))
        elif self.mask_mode in ['mean_dropout', 'sample_dropout']:
            self.masked_layer = MaskedInputLayer(use_random_sample=(self.mask_mode == 'sample_dropout'))

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers = None):
        batch_size = comp.size(0)
        proc_bool_mean = torch.tensor(scalers[1].mean_ if scalers else np.zeros(N_PROC_BOOL), dtype=torch.float32, device=comp.device)
        proc_scalar_mean = torch.tensor(scalers[2].mean_ if scalers else np.zeros(N_PROC_SCALAR), dtype=torch.float32, device=comp.device)
        proc_bool = proc_bool.reshape(batch_size, -1)
        proc_scalar = proc_scalar.reshape(batch_size, -1)
        
        if self.mask_mode == 'learned':
            if proc_bool_mask is not None:
                proc_bool_mask = proc_bool_mask.reshape(batch_size, -1)
                proc_bool = proc_bool * proc_bool_mask + (1 - proc_bool_mask) * self.missing_proc_bool.expand(batch_size, -1)
            if proc_scalar_mask is not None:
                proc_scalar_mask = proc_scalar_mask.reshape(batch_size, -1)
                proc_scalar = proc_scalar * proc_scalar_mask + (1 - proc_scalar_mask) * self.missing_proc_scalar.expand(batch_size, -1)
        elif self.mask_mode in ['mean_dropout', 'sample_dropout']:
            proc_bool = proc_bool.reshape(batch_size, -1)
            proc_bool_mask_reshaped = proc_bool_mask.reshape(batch_size, -1) if proc_bool_mask is not None else None
            proc_bool = self.masked_layer(proc_bool, proc_bool_mask_reshaped, proc_bool_mean)
            proc_scalar = proc_scalar.reshape(batch_size, -1)
            proc_scalar_mask_reshaped = proc_scalar_mask.reshape(batch_size, -1) if proc_scalar_mask is not None else None
            proc_scalar = self.masked_layer(proc_scalar, proc_scalar_mask_reshaped, proc_scalar_mean)
        
        x = comp.reshape(-1, N_ELEM)
        if self.Pr:
            x = torch.cat([x, proc_bool.reshape(-1, N_PROC_BOOL), proc_scalar.reshape(-1, N_PROC_SCALAR)], dim=-1)
        if self.Ph:
            x = torch.cat([x, phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x
    def get_name(self):
        name = "ELM"
        if self.Pr:
            name += "_Pr"
        if self.Ph:
            name += "_Ph"
        name += f'_{self.mask_mode}'
        return name

class ELM_ElemFeat(nn.Module):
    def __init__(self, mask_mode = 'zero',):
        super(ELM_ElemFeat, self).__init__()
        self.mask_mode = mask_mode
        
        self.fc1 = nn.Linear(N_ELEM + N_ELEM_FEAT + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR, N_FC_NERON)
        self.fc2 = nn.Linear(N_FC_NERON, N_PROP)
        self.af = nn.LeakyReLU(LEAKY_RATE)

        if self.mask_mode == 'learned':
            self.missing_proc_bool = nn.Parameter(torch.zeros(N_PROC_BOOL))
            self.missing_proc_scalar = nn.Parameter(torch.zeros(N_PROC_SCALAR))
            self.missing_phase_scalar = nn.Parameter(torch.zeros(N_PHASE_SCALAR))
        elif self.mask_mode in ['mean_dropout', 'sample_dropout']:
            self.masked_layer = MaskedInputLayer(use_random_sample=(self.mask_mode == 'sample_dropout'))

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers = None):
        batch_size = comp.size(0)
        proc_bool_mean = torch.tensor(scalers[1].mean_ if scalers else np.zeros(N_PROC_BOOL), dtype=torch.float32, device=comp.device)
        proc_scalar_mean = torch.tensor(scalers[2].mean_ if scalers else np.zeros(N_PROC_SCALAR), dtype=torch.float32, device=comp.device)
        proc_bool = proc_bool.reshape(batch_size, -1)
        proc_scalar = proc_scalar.reshape(batch_size, -1)
        
        if self.mask_mode == 'learned':
            if proc_bool_mask is not None:
                proc_bool_mask = proc_bool_mask.reshape(batch_size, -1)
                proc_bool = proc_bool * proc_bool_mask + (1 - proc_bool_mask) * self.missing_proc_bool.expand(batch_size, -1)
            if proc_scalar_mask is not None:
                proc_scalar_mask = proc_scalar_mask.reshape(batch_size, -1)
                proc_scalar = proc_scalar * proc_scalar_mask + (1 - proc_scalar_mask) * self.missing_proc_scalar.expand(batch_size, -1)
        elif self.mask_mode in ['mean_dropout', 'sample_dropout']:
            proc_bool = proc_bool.reshape(batch_size, -1)
            proc_bool_mask_reshaped = proc_bool_mask.reshape(batch_size, -1) if proc_bool_mask is not None else None
            proc_bool = self.masked_layer(proc_bool, proc_bool_mask_reshaped, proc_bool_mean)
            proc_scalar = proc_scalar.reshape(batch_size, -1)
            proc_scalar_mask_reshaped = proc_scalar_mask.reshape(batch_size, -1) if proc_scalar_mask is not None else None
            proc_scalar = self.masked_layer(proc_scalar, proc_scalar_mask_reshaped, proc_scalar_mean)

        mef = torch.sum(comp.squeeze(-1).squeeze(1).unsqueeze(-1) * elem_feat.squeeze(1), dim=1)
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            mef.reshape(-1, N_ELEM_FEAT), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x
    def get_name(self):
        return f"ELM_ElemFeat_{self.mask_mode}"

class ELM_CNN(nn.Module):
    def __init__(self, mask_mode = 'zero',):
        super(ELM_CNN, self).__init__()
        self.mask_mode = mask_mode

        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        self._n_cnn_out = N_ELEM * N_ELEM_FEAT_P1
        self._n_in_fcnn = self._n_cnn_out + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = N_FC_NERON
        
        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, N_PROP)
        self.af = nn.LeakyReLU(LEAKY_RATE)

        if self.mask_mode == 'learned':
            self.missing_proc_bool = nn.Parameter(torch.zeros(N_PROC_BOOL))
            self.missing_proc_scalar = nn.Parameter(torch.zeros(N_PROC_SCALAR))
            self.missing_phase_scalar = nn.Parameter(torch.zeros(N_PHASE_SCALAR))
        elif self.mask_mode in ['mean_dropout', 'sample_dropout']:
            self.masked_layer = MaskedInputLayer(use_random_sample=(self.mask_mode == 'sample_dropout'))

        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers = None):
        batch_size = comp.size(0)
        proc_bool_mean = torch.tensor(scalers[1].mean_ if scalers else np.zeros(N_PROC_BOOL), dtype=torch.float32, device=comp.device)
        proc_scalar_mean = torch.tensor(scalers[2].mean_ if scalers else np.zeros(N_PROC_SCALAR), dtype=torch.float32, device=comp.device)
        proc_bool = proc_bool.reshape(batch_size, -1)
        proc_scalar = proc_scalar.reshape(batch_size, -1)
        
        if self.mask_mode == 'learned':
            if proc_bool_mask is not None:
                proc_bool_mask = proc_bool_mask.reshape(batch_size, -1)
                proc_bool = proc_bool * proc_bool_mask + (1 - proc_bool_mask) * self.missing_proc_bool.expand(batch_size, -1)
            if proc_scalar_mask is not None:
                proc_scalar_mask = proc_scalar_mask.reshape(batch_size, -1)
                proc_scalar = proc_scalar * proc_scalar_mask + (1 - proc_scalar_mask) * self.missing_proc_scalar.expand(batch_size, -1)
        elif self.mask_mode in ['mean_dropout', 'sample_dropout']:
            proc_bool = proc_bool.reshape(batch_size, -1)
            proc_bool_mask_reshaped = proc_bool_mask.reshape(batch_size, -1) if proc_bool_mask is not None else None
            proc_bool = self.masked_layer(proc_bool, proc_bool_mask_reshaped, proc_bool_mean)
            proc_scalar = proc_scalar.reshape(batch_size, -1)
            proc_scalar_mask_reshaped = proc_scalar_mask.reshape(batch_size, -1) if proc_scalar_mask is not None else None
            proc_scalar = self.masked_layer(proc_scalar, proc_scalar_mask_reshaped, proc_scalar_mean)
            
        x = torch.cat([comp, elem_feat], dim=-1)
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
    def get_name(self):
        return f"ELM_CNN_{self.mask_mode}"


class FCNN(nn.Module):
    def __init__(self, mask_mode = 'zero', branch_mode = 'None', elem_feat = 'None'):
        super(FCNN, self).__init__()
        self.mask_mode = mask_mode
        self.branch_mode = branch_mode
        self.elem_feat = elem_feat

        self._n_fcnn = N_FC_NERON

        if self.elem_feat == 'None':
            self._n_mid_fcnn = N_FC_NERON
            self.layer0 = nn.Sequential(
                nn.Linear(N_ELEM + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR, self._n_fcnn),
                nn.LeakyReLU(LEAKY_RATE),
                nn.BatchNorm1d(self._n_fcnn),
                nn.Dropout(DROP_RATE),
                nn.Linear(self._n_fcnn, self._n_fcnn),
                nn.LeakyReLU(LEAKY_RATE),
                nn.BatchNorm1d(self._n_fcnn)
            )
        elif self.elem_feat == 'Mean':
            self._n_mid_fcnn = N_FC_NERON
            self.layer0 = nn.Sequential(
                nn.Linear(N_ELEM + N_ELEM_FEAT + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR, self._n_fcnn),
                nn.LeakyReLU(LEAKY_RATE),
                nn.BatchNorm1d(self._n_fcnn),
                nn.Dropout(DROP_RATE),
                nn.Linear(self._n_fcnn, self._n_fcnn),
                nn.LeakyReLU(LEAKY_RATE),
                nn.BatchNorm1d(self._n_fcnn)
            )
        elif self.elem_feat == 'CNN':
            self._n_cnn_out = N_ELEM * N_ELEM_FEAT_P1
            self._n_mid_fcnn = self._n_cnn_out + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
            self._kernel_size = (1, N_ELEM_FEAT_P1)
            self.CNN0 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size),
                nn.LeakyReLU(LEAKY_RATE),
                nn.BatchNorm2d(N_ELEM_FEAT_P1),
                nn.Dropout2d(DROP_RATE),
            )
            self.CNN1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size),
                nn.LeakyReLU(LEAKY_RATE),
                nn.BatchNorm2d(N_ELEM_FEAT_P1),
            )

        if self.branch_mode == 'None':
            self.layer1 = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_fcnn),
                nn.LeakyReLU(LEAKY_RATE),
                nn.BatchNorm1d(self._n_fcnn),
                nn.Linear(self._n_fcnn, self._n_fcnn),
                nn.LeakyReLU(LEAKY_RATE),
                nn.BatchNorm1d(self._n_fcnn),
                nn.Linear(self._n_fcnn, N_PROP)
            )
        elif branch_mode == 'MSHBranched':
            self._n_branch = N_BRANCH_NERON
            self.layer_m = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_branch),
                nn.BatchNorm1d(self._n_branch),
                nn.Linear(self._n_branch, self._n_branch),
                nn.Linear(self._n_branch, 1)
            )
            self.layer_s = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_branch),
                nn.BatchNorm1d(self._n_branch),
                nn.Linear(self._n_branch, self._n_branch),
                nn.Linear(self._n_branch, 3)
            )
            self.layer_h = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_branch),
                nn.BatchNorm1d(self._n_branch),
                nn.Linear(self._n_branch, self._n_branch),
                nn.Linear(self._n_branch, 1)
            )
        elif branch_mode == 'FullyBranched':
            self._n_branch = N_BRANCH_NERON
            self.layer1 = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_branch),
                nn.BatchNorm1d(self._n_branch),
                nn.Linear(self._n_branch, self._n_branch),
                nn.Linear(self._n_branch, 1)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_branch),
                nn.BatchNorm1d(self._n_branch),
                nn.Linear(self._n_branch, self._n_branch),
                nn.Linear(self._n_branch, 1)
            )
            self.layer3 = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_branch),
                nn.BatchNorm1d(self._n_branch),
                nn.Linear(self._n_branch, self._n_branch),
                nn.Linear(self._n_branch, 1)
            )
            self.layer4 = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_branch),
                nn.BatchNorm1d(self._n_branch),
                nn.Linear(self._n_branch, self._n_branch),
                nn.Linear(self._n_branch, 1)
            )
            self.layer5 = nn.Sequential(
                nn.Linear(self._n_mid_fcnn, self._n_branch),
                nn.BatchNorm1d(self._n_branch),
                nn.Linear(self._n_branch, self._n_branch),
                nn.Linear(self._n_branch, 1)
            )

        if self.mask_mode == 'learned':
            self.missing_proc_bool = nn.Parameter(torch.zeros(N_PROC_BOOL))
            self.missing_proc_scalar = nn.Parameter(torch.zeros(N_PROC_SCALAR))
            self.missing_phase_scalar = nn.Parameter(torch.zeros(N_PHASE_SCALAR))
        elif self.mask_mode in ['mean_dropout', 'sample_dropout']:
            self.masked_layer = MaskedInputLayer(use_random_sample=(self.mask_mode == 'sample_dropout'))

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

    def forward(self, comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers = None):
        batch_size = comp.size(0)
        proc_bool_mean = torch.tensor(scalers[1].mean_ if scalers else np.zeros(N_PROC_BOOL), dtype=torch.float32, device=comp.device)
        proc_scalar_mean = torch.tensor(scalers[2].mean_ if scalers else np.zeros(N_PROC_SCALAR), dtype=torch.float32, device=comp.device)
        proc_bool = proc_bool.reshape(batch_size, -1)
        proc_scalar = proc_scalar.reshape(batch_size, -1)
        
        if self.mask_mode == 'learned':
            if proc_bool_mask is not None:
                proc_bool_mask = proc_bool_mask.reshape(batch_size, -1)
                proc_bool = proc_bool * proc_bool_mask + (1 - proc_bool_mask) * self.missing_proc_bool.expand(batch_size, -1)
            if proc_scalar_mask is not None:
                proc_scalar_mask = proc_scalar_mask.reshape(batch_size, -1)
                proc_scalar = proc_scalar * proc_scalar_mask + (1 - proc_scalar_mask) * self.missing_proc_scalar.expand(batch_size, -1)
        elif self.mask_mode in ['mean_dropout', 'sample_dropout']:
            proc_bool = proc_bool.reshape(batch_size, -1)
            proc_bool_mask_reshaped = proc_bool_mask.reshape(batch_size, -1) if proc_bool_mask is not None else None
            proc_bool = self.masked_layer(proc_bool, proc_bool_mask_reshaped, proc_bool_mean)
            proc_scalar = proc_scalar.reshape(batch_size, -1)
            proc_scalar_mask_reshaped = proc_scalar_mask.reshape(batch_size, -1) if proc_scalar_mask is not None else None
            proc_scalar = self.masked_layer(proc_scalar, proc_scalar_mask_reshaped, proc_scalar_mean)
        
        if self.elem_feat == 'None':
            x = torch.cat([comp.reshape(-1, N_ELEM), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
            x = self.layer0(x)
        elif self.elem_feat == 'Mean':
            mef = torch.sum(comp.squeeze(-1).squeeze(1).unsqueeze(-1) * elem_feat.squeeze(1), dim=1)
            x = torch.cat([comp.reshape(-1, N_ELEM), 
                mef.reshape(-1, N_ELEM_FEAT), 
                proc_bool.reshape(-1, N_PROC_BOOL), 
                proc_scalar.reshape(-1, N_PROC_SCALAR),
                phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
            x = self.layer0(x)
        elif self.elem_feat == 'CNN':
            x = torch.cat([comp, elem_feat], dim=-1)
            residual = x
            x = self.CNN0(x)
            x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
            x = self.CNN1(x)
            x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
            x = x + residual
            x = x.view(-1, self._n_cnn_out)
            x = torch.cat([
                x,
                proc_bool.reshape(-1, N_PROC_BOOL),
                proc_scalar.reshape(-1, N_PROC_SCALAR),
                phase_scalar.reshape(-1, N_PHASE_SCALAR)
            ], dim=-1)

        if self.branch_mode == 'None':
            x = self.layer1(x)
        elif self.branch_mode == 'MSHBranched':
            m = self.layer_m(x)
            s = self.layer_s(x)
            h = self.layer_h(x)
            x = torch.cat([m, s, h], dim=-1)
        elif self.branch_mode == 'FullyBranched':
            x1 = self.layer1(x)
            x2 = self.layer2(x)
            x3 = self.layer3(x)
            x4 = self.layer4(x)
            x5 = self.layer5(x)
            x = torch.cat([x1, x2, x3, x4, x5], dim=-1)

        return x
    def get_name(self):
        name = "FCNN"
        if self.elem_feat != 'None':
            name += f'_{self.elem_feat}'
        if self.branch_mode != 'None':
            name += f'_{self.branch_mode}'
        name += f'_{self.mask_mode}'
        return name


class Attention(nn.Module):
    def __init__(self, mask_mode = 'zero', branch_mode = 'None'):
        super(Attention, self).__init__()
        self.mask_mode = mask_mode
        self.branch_mode = branch_mode

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
            nn.LeakyReLU(LEAKY_RATE),
            nn.Dropout(DROP_RATE),
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(LEAKY_RATE),
        )

        self.branch_dim = N_BRANCH_NERON
        if branch_mode == 'None':
            self.layer1 = self._make_head(self.branch_dim, N_PROP)
        elif branch_mode == 'MSHBranched':
            self.layer_m = self._make_head(self.branch_dim, 1)
            self.layer_s = self._make_head(self.branch_dim, 3)
            self.layer_h = self._make_head(self.branch_dim, 1)
        elif branch_mode == 'FullyBranched':
            self.layer1 = self._make_head(self.branch_dim, 1)
            self.layer2 = self._make_head(self.branch_dim, 1)
            self.layer3 = self._make_head(self.branch_dim, 1)
            self.layer4 = self._make_head(self.branch_dim, 1)
            self.layer5 = self._make_head(self.branch_dim, 1)

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

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

    def forward(self, comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers = None):
        batch_size = comp.size(0)
        
        comp_sq = comp.squeeze(-1).squeeze(1)
        feat_sq = elem_feat.squeeze(1)
        
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
        if self.branch_mode == 'None':
            x = self.layer1(x)
        elif self.branch_mode == 'MSHBranched':
            m = self.layer_m(x)
            s = self.layer_s(x)
            h = self.layer_h(x)
            x = torch.cat([m, s, h], dim=-1)
        elif self.branch_mode == 'FullyBranched':
            x1 = self.layer1(x)
            x2 = self.layer2(x)
            x3 = self.layer3(x)
            x4 = self.layer4(x)
            x5 = self.layer5(x)
            x = torch.cat([x1, x2, x3, x4, x5], dim=-1)

        return x
    
    def get_attention_weights(self, comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers = None):
        self.eval()
        with torch.no_grad():
            comp_sq = comp.squeeze(-1).squeeze(1)
            feat_sq = elem_feat.squeeze(1)
            
            elem_emb = self.elem_encoder(feat_sq)
            gate = self.comp_gate(comp_sq.unsqueeze(-1))
            elem_emb_gated = elem_emb * gate
            
            _, attn_weights = self.attention(
                elem_emb_gated, elem_emb_gated, elem_emb_gated
            )
            
        return attn_weights, gate.squeeze(-1)
    
    def get_name(self):
        name = "Attention"
        if self.branch_mode != 'None':
            name += f'_{self.branch_mode}'
        return name


class TiAlloyNet(nn.Module):
    def __init__(self, mask_mode = 'zero', branch_mode = 'None', Pr = True, Ph = True):
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

    def forward(self, comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers = None):
        x = torch.cat([comp, elem_feat], dim=-1)
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
    def get_name(self):
        name = "TiAlloyNet"
        return name

def MODEL_LIST(mask_mode: str) -> list:
    model_list = []
    ELM_list = [ELM(mask_mode = mask_mode, Pr = Pr, Ph = Ph).to(device)\
         for Pr in [True, False] for Ph in [True, False]]
    ELM_list.append(ELM_ElemFeat(mask_mode = mask_mode).to(device))
    ELM_list.append(ELM_CNN(mask_mode = mask_mode).to(device))
    model_list.extend(ELM_list)

    FCNN_list = [FCNN(mask_mode = mask_mode, branch_mode = Branch_mode, elem_feat = elem_feat).to(device)\
         for Branch_mode in ['None', 'MSHBranched', 'FullyBranched']\
         for elem_feat in ['None', 'Mean', 'CNN']]
    model_list.extend(FCNN_list)
    Attention_list = [Attention(mask_mode = mask_mode, branch_mode = Branch_mode).to(device)\
         for Branch_mode in ['None', 'MSHBranched', 'FullyBranched']]
    model_list.extend(Attention_list)
    # model_list.append(TiAlloyNet(mask_mode = mask_mode).to(device))

    return model_list

if __name__ == '__main__':
    _batch_size = 8
    test_input = (
        torch.ones((_batch_size, 1, N_ELEM, 1)).to(device),
        torch.ones((_batch_size, 1, N_ELEM, N_ELEM_FEAT)).to(device),
        torch.ones((_batch_size, 1, N_PROC_BOOL, 1)).to(device),
        torch.ones((_batch_size, 1, N_PROC_SCALAR, 1)).to(device),
        torch.ones((_batch_size, 1, N_PHASE_SCALAR, 1)).to(device),
        torch.ones((_batch_size, 1, N_PROC_BOOL, 1)).to(device),
        torch.ones((_batch_size, 1, N_PROC_SCALAR, 1)).to(device),
    )
    mask_modes = ['zero', 'learned', 'mean_dropout', 'sample_dropout']
    for mask_mode in mask_modes:
        model_list = MODEL_LIST(mask_mode)
        for model in model_list:
            print(f"{model.get_name()} {list(model(*test_input).size())}")

    
