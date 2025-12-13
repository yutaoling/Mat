import random
from typing import Callable, Tuple
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from model_env import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC, N_PROP
from model_env import CnnDnnModel, device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
set_seed(0) # default 0

seeds = np.random.randint(0, 9999, (9999, ))

PROP=['ym', 'ys', 'uts', 'el', 'hv']
PROP_LABELS = {'ym':'YM(GPa)', 'ys':'YS(MPa)', 'uts':'UTS(MPa)',
               'el':'El(%)', 'dar':'DAR(%)', 'hv':'HV'}

def load_data():
    # Load the default dataset
    data = pd.read_excel('data\\Ti_dataset.xlsx')
    
    # composition labels
    comp_labels = ['C(at%)', 'N(at%)', 'O(at%)', 'Al(at%)', 'Si(at%)', 'Sc(at%)',
                   'Ti(at%)', 'V(at%)', 'Cr(at%)', 'Mn(at%)', 'Fe(at%)',
                   'Ni(at%)', 'Cu(at%)', 'Zr(at%)', 'Nb(at%)', 'Mo(at%)', 'Sn(at%)',
                   'Hf(at%)', 'Ta(at%)', 'W(at%)']
                        
    # processing condition labels
    proc_labels = ['Hom_Temp(K)']

    # property labels
    # YM(GPa), YS(MPa), UTS(MPa), El(%), HV
    prop_labels = [PROP_LABELS[pl] for pl in PROP]
    # data = data.dropna(subset=prop_labels)

    comp_data = data[comp_labels].to_numpy()
    proc_data = data[proc_labels].to_numpy()
    prop_data = data[prop_labels].to_numpy()

    elem_feature = pd.read_excel('data\\elemental_features.xlsx')
    elem_feature = elem_feature[
        ['C', 'N', 'O', 'Al', 'Si', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni',
         'Cu', 'Zr', 'Nb', 'Mo', 'Sn', 'Hf', 'Ta', 'W']
    ].to_numpy()  # transpose: column for each elemental feature, row for each element 

    # (num_samples, num_elements), (num_samples, num_proc), (num_samples, num_prop), (num_elements, num_elem_features,)
    return comp_data, proc_data, prop_data, elem_feature

def fit_transform(data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]):
    '''fit and transform the data'''
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    comp_data_scaler, proc_data_scaler, prop_data_scaler, elem_feature_scaler = \
        [StandardScaler() for _ in range(len(data_tuple))]
    
    comp_data = comp_data_scaler.fit_transform(comp_data)
    proc_data = proc_data_scaler.fit_transform(proc_data)
    prop_data = prop_data_scaler.fit_transform(prop_data)
    ''' 
        input elem_feature:     (num_elem_features, num_elements, ), as defined in the EXCEL file
        output elem_feature:    (num_elements, num_elem_features, )
        however sklearn scaler works colum-wise,
        should calculate the mu and sigma of element features (say, VEC) for diff elements,
        so transpose the elem_feature
    '''
    elem_feature = elem_feature_scaler.fit_transform(elem_feature.T)

    # return the data and the scalers
    return (
        (comp_data, proc_data, prop_data, elem_feature,),
        (comp_data_scaler, proc_data_scaler, prop_data_scaler, elem_feature_scaler,),
    )

class CustomDataset(Dataset):
    ''' store comp, proc, prop data '''
    def __init__(self, 
                 data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray,],
                 scaler: TransformerMixin): # TODO deprecate scaler
        self.data_tuple = data_tuple
        self.scaler = scaler

        self.comp = self.data_tuple[0]
        self.proc = self.data_tuple[1]
        self.prop = self.data_tuple[2]

        self.mask = ~np.isnan(self.prop)
        
    def __len__(self):
        return len(self.comp)
    
    def __getitem__(self, idx):
        _comp = self.comp[idx]
        _proc = self.proc[idx]
        _prop = self.prop[idx]
        _mask = self.mask[idx]

        _prop = np.nan_to_num(_prop, nan=-1)
        
        return _comp, _proc, _prop, _mask

def get_dataloader(data_tuple, batch_size = 16) -> DataLoader:
    ''' 
        get the dataloader

        input:
            (comp_data, proc_data, prop_data, elem_feature)
            elem_feature: (num_elem_features, num_elements)
    '''
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    dataset = CustomDataset((comp_data, proc_data, prop_data,), None)

    # target elem_feature: (batch_size, 1, number_of_elements, number_of_elemental_features)
    _elem_feature_tensor = torch.tensor(elem_feature, dtype=torch.float32).reshape(1, 1, *(elem_feature.shape))

    def _collate_fn(batch):
        comp, proc, prop, mask = zip(*batch)
        comp = torch.tensor(np.vstack(comp), dtype=torch.float32).reshape(-1, 1, comp_data.shape[-1], 1)
        proc = torch.tensor(np.vstack(proc), dtype=torch.float32).reshape(-1, 1, proc_data.shape[-1], 1)
        prop = torch.tensor(np.vstack(prop), dtype=torch.float32).reshape(-1, 1, prop_data.shape[-1], 1)
        mask = torch.tensor(np.vstack(mask), dtype=torch.float32).reshape(-1, 1, prop_data.shape[-1], 1)

        _elem_feature_tensor_clone = _elem_feature_tensor.expand(len(comp), 1, *(elem_feature.shape)).clone().detach()
        _elem_feature_tensor_clone.requires_grad_(False)

        comp=comp.to(device)
        proc=proc.to(device)
        prop=prop.to(device)
        mask=mask.to(device)
        _elem_feature_tensor_clone=_elem_feature_tensor_clone.to(device)

        return comp, proc, prop, mask, _elem_feature_tensor_clone

    return DataLoader(dataset, batch_size = batch_size, collate_fn = _collate_fn, shuffle = True)

def MaskedMSELoss(out, prop, mask):
    mse_loss = nn.functional.mse_loss(out, prop, reduction='none')
    masked_loss = mse_loss * mask
    num_valid = mask.sum().clamp(min=1e-6)
    mean_masked_loss = masked_loss.sum()/num_valid
    return mean_masked_loss

def train_validate_split(data_tuple, ratio_tuple = (0.95, 0.04, 0.01)):
    ''' 
        split the data into train, validate_1 and validate_2 set
    '''
    _random_seed = next(iter(seeds))
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    _ratio_1 = sum(ratio_tuple[1:]) / sum(ratio_tuple)
    comp_train, comp_tmp, proc_train, proc_tmp, prop_train, prop_tmp = \
        train_test_split(comp_data, proc_data, prop_data, test_size = _ratio_1, random_state = _random_seed)
    _ratio_2 = ratio_tuple[2] / sum(ratio_tuple[1:])
    comp_val_1, comp_val_2, proc_val_1, proc_val_2, prop_val_1, prop_val_2 = \
        train_test_split(comp_tmp, proc_tmp, prop_tmp, test_size = _ratio_2, random_state = _random_seed)
    
    return (comp_train, proc_train, prop_train, elem_feature,), \
            (comp_val_1, proc_val_1, prop_val_1, elem_feature,), \
            (comp_val_2, proc_val_2, prop_val_2, elem_feature,)


def validate(model: CnnDnnModel, data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]) -> float:
    ''' calculate the R2 score of the model on the validate set '''
    model.to(device)
    model.eval()
    dl = get_dataloader(data_tuple, len(data_tuple[0]))
    comp, proc, prop, mask, elem_t = next(iter(dl))
    comp = comp.to(device)
    proc = proc.to(device)
    prop = prop.to(device)
    mask = mask.to(device)
    elem_t = elem_t.to(device)
    out = model(comp, elem_t, proc).detach()
    prop = prop.reshape(*(out.shape))
    mask = mask.reshape(*(out.shape))
    mse_loss = MaskedMSELoss(out, prop, mask)
    return mse_loss.item()

def validate_a_model(num_training_epochs = 2000,
                     batch_size = 16,
                     save_path = None,):
    ''' util func for training_epoch_num validation '''
    model = CnnDnnModel().to(device)
    d = load_data()
    d, scalers = fit_transform(d)

    train_d, val_d_1, val_d_2 = train_validate_split(d, (0.9, 0.05, 0.05))
    loss_fn = torch.nn.MSELoss()
    dl = get_dataloader(train_d, batch_size)
    # train one epoch
    epoch_log_buffer = []
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for comp, proc, prop, mask, elem_t in dl:
            # forward pass
            comp = comp.to(device)
            proc = proc.to(device)
            prop = prop.to(device)
            elem_t = elem_t.to(device)
            out = model(comp, elem_t, proc)
            l = MaskedMSELoss(out, prop.reshape(*(out.shape)), mask.reshape(*(out.shape)))

            # backward pass
            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            _batch_loss_buffer.append(l.item())
        
        # model.eval()
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        val_1_r2 = validate(model, val_d_1)
        val_2_r2 = validate(model, val_d_2)
        epoch_log_buffer.append((epoch, _batch_mean_loss, val_1_r2, val_2_r2))
        if not epoch % 25:
            print(epoch, _batch_mean_loss, val_1_r2, val_2_r2)
    
    if save_path:
        np.savetxt(
            save_path,
            np.array(epoch_log_buffer),
            fmt = '%.6f',
            delimiter = '\t',
        )
    
    return model, d, scalers

def train_a_model(num_training_epochs = 1000,
                    batch_size = 16,
                    save_path = None,):
    ''' train a model '''
    model = CnnDnnModel().to(device)
    d = load_data()
    d, scalers = fit_transform(d)

    train_d = d
    loss_fn = torch.nn.MSELoss()
    dl = get_dataloader(train_d, batch_size)
    # train one epoch
    epoch_log_buffer = []
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for comp, proc, prop, mask, elem_t in dl:
            # forward pass
            out = model(comp, elem_t, proc)
            l = MaskedMSELoss(out, prop.reshape(*(out.shape)), mask.reshape(*(out.shape)))

            # backward pass
            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            _batch_loss_buffer.append(l.item())
        
        # model.eval()
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        epoch_log_buffer.append((epoch, _batch_mean_loss))
        if epoch % 25 == 0: 
            print(epoch, _batch_mean_loss)
    
    if save_path:
        np.savetxt(
            save_path,
            np.array(epoch_log_buffer),
            fmt = '%.6f',
            delimiter = '\t',
        )
    
    return model, d, scalers

def get_model(default_model_pth = 'model.pth',
              default_data_pth = 'data.pth',
              resume = False,
              save_path=None,):

    if resume:
        model = CnnDnnModel().to(device)
        model.load_state_dict(torch.load(default_model_pth, map_location=device))
        d, scalers = joblib.load(default_data_pth)
    else:
        model, d, scalers=train_a_model(save_path=save_path, num_training_epochs=3000)
        torch.save(model.state_dict(), default_model_pth)
        joblib.dump((d, scalers), default_data_pth)
    
    return model, d, scalers

if __name__ == '__main__':
    get_model(f'model_multi.pth',f'data_multi.pth',resume=False,save_path='model_multi_train_err_log.txt')
    # validate_a_model(num_training_epochs=3000, save_path='model_multi_valid_log_01simple.txt')
