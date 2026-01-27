import numpy as np
import torch
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
LEARNING_RATE = 5e-4

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Baseline_Model(nn.Module):
    def __init__(self):
        super(Baseline_Model, self).__init__()
        
        self.fc1 = nn.Linear(N_ELEM + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR, 128)
        self.fc2 = nn.Linear(128, 5)
        self.af = nn.LeakyReLU(0.2)
        
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

class Baseline_Comp_Model(nn.Module):
    def __init__(self):
        super(Baseline_Comp_Model, self).__init__()
        
        self.fc1 = nn.Linear(N_ELEM, 128)
        self.fc2 = nn.Linear(128, 5)
        self.af = nn.LeakyReLU(0.2)
        
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

class Baseline_CompPhase_Model(nn.Module):
    def __init__(self):
        super(Baseline_CompPhase_Model, self).__init__()
        
        self.fc1 = nn.Linear(N_ELEM + N_PHASE_SCALAR, 128)
        self.fc2 = nn.Linear(128, 5)
        self.af = nn.LeakyReLU(0.2)
        
        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        x = torch.cat([comp.reshape(-1, N_ELEM), phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        return x


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

class FCNN_Model(nn.Module):
    def __init__(self):
        super(FCNN_Model, self).__init__()

        self._n_in_fcnn = N_ELEM + N_ELEM_FEAT + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = 128
        self._n_branch = 64
        self._n_hv = 32
        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)
        self.fc3 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn3 = nn.BatchNorm1d(self._n_fcnn)

        self.fc_ym1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_ym = nn.BatchNorm1d(self._n_branch)
        self.fc_ym2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ym = nn.Linear(self._n_branch, 1)

        self.fc_s1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_s1 = nn.BatchNorm1d(self._n_branch)
        self.fc_s2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_s = nn.Linear(self._n_branch, 3)

        self.fc_hv1 = nn.Linear(self._n_fcnn, self._n_hv)
        self.bn_hv = nn.BatchNorm1d(self._n_hv)
        self.fc_hv2 = nn.Linear(self._n_hv, self._n_hv)
        self.out_hv = nn.Linear(self._n_hv, 1)

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

    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):
        mef = torch.sum(comp.squeeze(-1).squeeze(1).unsqueeze(-1) * elem_feature.squeeze(1), dim=1)
        x = torch.cat([comp.reshape(-1, N_ELEM), 
            mef.reshape(-1, N_ELEM_FEAT), 
            proc_bool.reshape(-1, N_PROC_BOOL), 
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        res = x
        x = self.af(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.af(self.bn3(self.fc3(x)) + res)

        ym = self.af(self.bn_ym(self.fc_ym1(x)))
        ym = self.af(self.fc_ym2(ym))
        ym = self.out_ym(ym)

        s = self.af(self.bn_s1(self.fc_s1(x)))
        s = self.af(self.fc_s2(s))
        s = self.out_s(s)

        hv = self.af(self.bn_hv(self.fc_hv1(x)))
        hv = self.af(self.fc_hv2(hv))
        hv = self.out_hv(hv)
        
        x = torch.cat([ym, s, hv], dim=-1)
        return x

class CNN_Branched_Model(nn.Module):

    def __init__(self):
        super(CNN_Branched_Model, self).__init__()
        
        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=N_ELEM_FEAT_P1, kernel_size=self._kernel_size)
        self.bn_conv2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        self._n_cnn_out = N_ELEM * N_ELEM_FEAT_P1  # 19 * 31 = 589
        self._n_in_fcnn = self._n_cnn_out + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR
        self._n_fcnn = 128
        self._n_branch = 64
        self._n_hv = 32
        
        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)
        self.fc3 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn3 = nn.BatchNorm1d(self._n_fcnn)
        
        self.fc_ym1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_ym = nn.BatchNorm1d(self._n_branch)
        self.fc_ym2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ym = nn.Linear(self._n_branch, 1)
        
        self.fc_s1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_s1 = nn.BatchNorm1d(self._n_branch)
        self.fc_s2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_s = nn.Linear(self._n_branch, 3)
        
        self.fc_hv1 = nn.Linear(self._n_fcnn, self._n_hv)
        self.bn_hv = nn.BatchNorm1d(self._n_hv)
        self.fc_hv2 = nn.Linear(self._n_hv, self._n_hv)
        self.out_hv = nn.Linear(self._n_hv, 1)
        
        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)
        
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
        x = torch.cat([comp, elem_feature], dim=-1)  # (batch, 1, N_ELEM, N_ELEM_FEAT_P1)
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
        res = x
        x = self.af(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.af(self.bn3(self.fc3(x)) + res)
        
        ym = self.af(self.bn_ym(self.fc_ym1(x)))
        ym = self.af(self.fc_ym2(ym))
        ym = self.out_ym(ym)
        
        s = self.af(self.bn_s1(self.fc_s1(x)))
        s = self.af(self.fc_s2(s))
        s = self.out_s(s)
        
        hv = self.af(self.bn_hv(self.fc_hv1(x)))
        hv = self.af(self.fc_hv2(hv))
        hv = self.out_hv(hv)
        
        out = torch.cat([ym, s, hv], dim=-1)
        return out

class FCNN_NoElemFeat_Model(nn.Module):
    def __init__(self):
        super(FCNN_NoElemFeat_Model, self).__init__()
        
        # 输入维度：成分 + 工艺参数（无元素特征）
        self._n_in_fcnn = N_ELEM + N_PROC_BOOL + N_PROC_SCALAR + N_PHASE_SCALAR  # 19 + 8 + 6 + 3 = 36
        self._n_fcnn = 128
        self._n_branch = 64
        self._n_hv = 32
        
        self.fc1 = nn.Linear(self._n_in_fcnn, self._n_fcnn)
        self.bn1 = nn.BatchNorm1d(self._n_fcnn)
        self.fc2 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn2 = nn.BatchNorm1d(self._n_fcnn)
        self.fc3 = nn.Linear(self._n_fcnn, self._n_fcnn)
        self.bn3 = nn.BatchNorm1d(self._n_fcnn)
        
        self.fc_ym1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_ym = nn.BatchNorm1d(self._n_branch)
        self.fc_ym2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_ym = nn.Linear(self._n_branch, 1)
        
        self.fc_s1 = nn.Linear(self._n_fcnn, self._n_branch)
        self.bn_s1 = nn.BatchNorm1d(self._n_branch)
        self.fc_s2 = nn.Linear(self._n_branch, self._n_branch)
        self.out_s = nn.Linear(self._n_branch, 3)
        
        self.fc_hv1 = nn.Linear(self._n_fcnn, self._n_hv)
        self.bn_hv = nn.BatchNorm1d(self._n_hv)
        self.fc_hv2 = nn.Linear(self._n_hv, self._n_hv)
        self.out_hv = nn.Linear(self._n_hv, 1)
        
        self.af = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)
        
        self.reset_parameters()
        
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    m.weight.data.fill_(1.)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, comp, elem_feature, proc_bool, proc_scalar, phase_scalar):

        x = torch.cat([
            comp.reshape(-1, N_ELEM),
            proc_bool.reshape(-1, N_PROC_BOOL),
            proc_scalar.reshape(-1, N_PROC_SCALAR),
            phase_scalar.reshape(-1, N_PHASE_SCALAR)
        ], dim=-1)
        
        x = self.af(self.bn1(self.fc1(x)))
        res = x
        x = self.af(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.af(self.bn3(self.fc3(x)) + res)
        
        ym = self.af(self.bn_ym(self.fc_ym1(x)))
        ym = self.af(self.fc_ym2(ym))
        ym = self.out_ym(ym)
        
        s = self.af(self.bn_s1(self.fc_s1(x)))
        s = self.af(self.fc_s2(s))
        s = self.out_s(s)
        
        hv = self.af(self.bn_hv(self.fc_hv1(x)))
        hv = self.af(self.fc_hv2(hv))
        hv = self.out_hv(hv)
        
        out = torch.cat([ym, s, hv], dim=-1)
        return out

class Attention_Model(nn.Module):
    def __init__(self):
        super(Attention_Model, self).__init__()

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
        self.global_hidden_dim = 128

        self.fc_main = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),  # 降低 dropout
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.branch_dim = 64

        self.head_ym = self._make_head(self.branch_dim, 1)
        self.head_s = self._make_head(self.branch_dim, 3)
        self.head_hv = self._make_head(32, 1)

        self.lr = LEARNING_RATE        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)  # 降低 weight_decay
        
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
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
        )  # (batch, N_ELEM, attn_hidden_dim)
        
        attn_out = self.attn_proj(attn_out)
        attn_emb = torch.sum(comp_weights * attn_out, dim=1)
        
        proc_phase = torch.cat([
            proc_bool.reshape(batch_size, -1), 
            proc_scalar.reshape(batch_size, -1),
            phase_scalar.reshape(batch_size, -1)
        ], dim=-1)  # (batch, proc_phase_dim)
        
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



if __name__ == '__main__':
    _batch_size = 8
    
    test_input = (
        torch.ones((_batch_size, 1, N_ELEM, 1)).to(device),
        torch.ones((_batch_size, 1, N_ELEM, N_ELEM_FEAT)).to(device),
        torch.ones((_batch_size, 1, N_PROC_BOOL, 1)).to(device),
        torch.ones((_batch_size, 1, N_PROC_SCALAR, 1)).to(device),
        torch.ones((_batch_size, 1, N_PHASE_SCALAR, 1)).to(device)
    )
    
    model_baseline = Baseline_Model().to(device)
    print(f"{model_baseline(*test_input).size()}")
    
    model_baseline_comp_only = Baseline_Comp_Model().to(device)
    print(f"{model_baseline_comp_only(*test_input).size()}")
    
    model_baseline_comp_phase = Baseline_CompPhase_Model().to(device)
    print(f"{model_baseline_comp_phase(*test_input).size()}")
    
    model_fcnn = FCNN_Model().to(device)
    print(f"{model_fcnn(*test_input).size()}")
    
    model_cnn_branched = CNN_Branched_Model().to(device)
    print(f"{model_cnn_branched(*test_input).size()}")
    
    model_no_elem_feat = FCNN_NoElemFeat_Model().to(device)
    print(f"{model_no_elem_feat(*test_input).size()}")
    
    model_attn = Attention_Model().to(device)
    print(f"{model_attn(*test_input).size()}")