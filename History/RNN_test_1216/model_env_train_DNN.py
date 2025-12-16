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

from model_env_DNN import COMP, PROC_BOOL, PROC_SCALAR, PROP, PROP_LABELS
from model_env_DNN import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC_BOOL, N_PROC_SCALAR, N_PROP
from model_env_DNN import CnnDnnModel, device

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


def load_data():
    # Load the default dataset
    data = pd.read_csv('data\\Ti_dataset.csv')
    data.loc[range(21, 24), 'Activated'] = 0
    # They are testset
    data = data[data['Activated'] == 1]

    no_labels = ['No']
    # composition labels
    comp_labels = COMP
                        
    # processing condition labels
    proc_bool_labels=PROC_BOOL
    proc_scalar_labels=PROC_SCALAR

    # property labels
    # YM(GPa), YS(MPa), UTS(MPa), El(%), HV
    prop_labels = PROP#[PROP_LABELS[pl] for pl in PROP]

    # data = data.dropna(subset=prop_labels)
    # Normalize whitespace-only cells to NaN and coerce numeric columns to numbers
    # data = data.replace(r'^\s*$', 0, regex=True)
    # data = data.replace(np.nan, 0, regex=True)
    cols_to_numeric = comp_labels + proc_bool_labels + proc_scalar_labels + prop_labels
    # Some of these columns may not exist in every dataset; filter to existing ones
    cols_to_numeric = [c for c in cols_to_numeric if c in data.columns]
    if cols_to_numeric:
        data[cols_to_numeric] = data[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

    no_data = data[no_labels].to_numpy()
    comp_data = data[comp_labels].to_numpy()
    proc_bool_data = data[proc_bool_labels].to_numpy()
    proc_scalar_data = data[proc_scalar_labels].to_numpy()
    prop_data = data[prop_labels].to_numpy()


    elem_feature = pd.read_excel('data\\elemental_features.xlsx')
    elem_feature = elem_feature[comp_labels].to_numpy()  # transpose: column for each elemental feature, row for each element 
    # (num_samples, num_elements), (num_samples, num_proc), (num_samples, num_prop), (num_elements, num_elem_features,)
    return no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data, elem_feature

def fit_transform(data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]):
    '''fit and transform the data'''
    no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data, elem_feature = data_tuple
    comp_data_scaler, proc_scalar_scaler, prop_data_scaler, elem_feature_scaler = \
        [StandardScaler() for _ in range(len(data_tuple)-2)]
    
    comp_data = comp_data_scaler.fit_transform(comp_data)
    proc_scalar_data = proc_scalar_scaler.fit_transform(proc_scalar_data)
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
        (no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data, elem_feature,),
        (comp_data_scaler, proc_scalar_scaler, prop_data_scaler, elem_feature_scaler,),
    )

class CustomDataset(Dataset):
    ''' store comp, proc, prop data '''
    def __init__(self, 
                 data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,],
                 scaler: TransformerMixin): # TODO deprecate scaler
        self.data_tuple = data_tuple
        self.scaler = scaler

        self.no = self.data_tuple[0]
        self.comp = self.data_tuple[1]
        self.proc_bool = self.data_tuple[2]
        self.proc_scalar = self.data_tuple[3]
        self.prop = self.data_tuple[4]

        self.mask = ~np.isnan(self.prop)
        
    def __len__(self):
        return len(self.comp)
    
    def __getitem__(self, idx):
        _no = self.no[idx]
        _comp = self.comp[idx]
        _proc_bool = self.proc_bool[idx]
        _proc_scalar = self.proc_scalar[idx]
        _prop = self.prop[idx]
        _mask = self.mask[idx]

        _prop = np.nan_to_num(_prop, nan=-1)
        
        return _no, _comp, _proc_bool, _proc_scalar, _prop, _mask

def get_dataloader(data_tuple, batch_size = 16) -> DataLoader:
    ''' 
        get the dataloader

        input:
            (comp_data, proc_data, prop_data, elem_feature)
            elem_feature: (num_elem_features, num_elements)
    '''
    no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data, elem_feature = data_tuple
    dataset = CustomDataset((no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data,), None)

    # target elem_feature: (batch_size, 1, number_of_elements, number_of_elemental_features)
    _elem_feature_tensor = torch.tensor(elem_feature, dtype=torch.float32).reshape(1, 1, *(elem_feature.shape))

    def _collate_fn(batch):
        no, comp, proc_bool, proc_scalar, prop, mask = zip(*batch)
        no = torch.tensor(np.vstack(no), dtype=torch.float32).reshape(-1, 1, no_data.shape[-1], 1)
        comp = torch.tensor(np.vstack(comp), dtype=torch.float32).reshape(-1, 1, comp_data.shape[-1], 1)
        proc_bool = torch.tensor(np.vstack(proc_bool), dtype=torch.float32).reshape(-1, 1, proc_bool_data.shape[-1], 1)
        proc_scalar = torch.tensor(np.vstack(proc_scalar), dtype=torch.float32).reshape(-1, 1, proc_scalar_data.shape[-1], 1)
        prop = torch.tensor(np.vstack(prop), dtype=torch.float32).reshape(-1, 1, prop_data.shape[-1], 1)
        mask = torch.tensor(np.vstack(mask), dtype=torch.float32).reshape(-1, 1, prop_data.shape[-1], 1)

        _elem_feature_tensor_clone = _elem_feature_tensor.expand(len(comp), 1, *(elem_feature.shape)).clone().detach()
        _elem_feature_tensor_clone.requires_grad_(False)

        no=no.to(device)
        comp=comp.to(device)
        proc_bool=proc_bool.to(device)
        proc_scalar=proc_scalar.to(device)
        prop=prop.to(device)
        mask=mask.to(device)
        _elem_feature_tensor_clone=_elem_feature_tensor_clone.to(device)

        return no, comp, proc_bool, proc_scalar, prop, mask, _elem_feature_tensor_clone

    return DataLoader(dataset, batch_size = batch_size, collate_fn = _collate_fn, shuffle = True)

def MaskedMSELoss(out, prop, mask):
    mse_loss = nn.functional.mse_loss(out, prop, reduction='none')
    masked_loss = mse_loss * mask
    num_valid = mask.sum().clamp(min=1e-6)
    mean_masked_loss = masked_loss.sum()/num_valid
    return mean_masked_loss

def train_validate_split(data_tuple, ratio_tuple = (0.95, 0.05)):
    ''' 
        split the data into train, validate_1 and validate_2 set
    '''
    _random_seed = next(iter(seeds))
    no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data, elem_feature = data_tuple
    _ratio_1 = sum(ratio_tuple[1:]) / sum(ratio_tuple)
    no_train, no_val, comp_train, comp_val, proc_bool_train, proc_bool_val, proc_scalar_train, proc_scalar_val, prop_train, prop_val = \
        train_test_split(no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data, test_size = _ratio_1, random_state = 114514)# _random_seed
    return (no_train, comp_train, proc_bool_train, proc_scalar_train, prop_train, elem_feature,), \
            (no_val, comp_val, proc_bool_val, proc_scalar_val, prop_val, elem_feature,)

def train_validate_2_split(data_tuple, ratio_tuple = (0.95, 0.04, 0.01)):
    ''' 
        split the data into train, validate_1 and validate_2 set
    '''
    _random_seed = next(iter(seeds))
    no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data, elem_feature = data_tuple
    _ratio_1 = sum(ratio_tuple[1:]) / sum(ratio_tuple)
    no_train, no_tmp, comp_train, comp_tmp, proc_bool_train, proc_bool_tmp, proc_scalar_train, proc_scalar_tmp, prop_train, prop_tmp = \
        train_test_split(no_data, comp_data, proc_bool_data, proc_scalar_data, prop_data, test_size = _ratio_1, random_state = 114514)# _random_seed
    _ratio_2 = ratio_tuple[2] / sum(ratio_tuple[1:])
    no_val_1, no_val_2, comp_val_1, comp_val_2, proc_bool_val_1, proc_bool_val_2, proc_scalar_val_1, proc_scalar_val_2, prop_val_1, prop_val_2 = \
        train_test_split(no_tmp, comp_tmp, proc_bool_tmp, proc_scalar_tmp, prop_tmp, test_size = _ratio_2, random_state = 114514)# _random_seed
    return (no_train, comp_train, proc_bool_train, proc_scalar_train, prop_train, elem_feature,), \
            (no_val_1, comp_val_1, proc_bool_val_1, proc_scalar_val_1, prop_val_1, elem_feature,), \
            (no_val_2, comp_val_2, proc_bool_val_2, proc_scalar_val_2, prop_val_2, elem_feature,)

def validate(model: CnnDnnModel, data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]) -> float:
    ''' calculate the R2 score of the model on the validate set '''
    model.to(device)
    model.eval()
    dl = get_dataloader(data_tuple, len(data_tuple[0]))
    no, comp, proc_bool, proc_scalar, prop, mask, elem_t = next(iter(dl))

    no = no.to(device)
    comp = comp.to(device)
    proc_bool = proc_bool.to(device)
    proc_scalar = proc_scalar.to(device)
    prop = prop.to(device)
    mask = mask.to(device)
    elem_t = elem_t.to(device)
    out = model(comp, elem_t, proc_bool, proc_scalar).detach()
    prop = prop.reshape(*(out.shape))
    mask = mask.reshape(*(out.shape))
    mse_loss = MaskedMSELoss(out, prop, mask)
    return mse_loss.item()

def validate_a_model(num_training_epochs = 2000,
                     batch_size = 16,
                     save_path = None,):
    ''' util func for training_epoch_num validation '''
    model = CnnDnnModel().to(device)
    # d = load_data()
    # d, scalers = fit_transform(d)
    d, scalers = joblib.load('data_multi.pth')
    # train_d, val_d = train_validate_split(d, (0.9, 0.1))
    train_d, val_d, scalers = joblib.load('data_multi_divided.pth')
    loss_fn = torch.nn.MSELoss()
    dl = get_dataloader(train_d, batch_size)
    # train one epoch
    epoch_log_buffer = []
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for no, comp, proc_bool, proc_scalar, prop, mask, elem_t in dl:
            # forward pass
            no = no.to(device)
            comp = comp.to(device)
            proc_bool = proc_bool.to(device)
            proc_scalar = proc_scalar.to(device)
            prop = prop.to(device)
            elem_t = elem_t.to(device)
            out = model(comp, elem_t, proc_bool, proc_scalar)
            l = MaskedMSELoss(out, prop.reshape(*(out.shape)), mask.reshape(*(out.shape)))

            # backward pass
            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            _batch_loss_buffer.append(l.item())
        
        # model.eval()
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        val_r2 = validate(model, val_d)
        epoch_log_buffer.append((epoch, _batch_mean_loss, val_r2))
        if not epoch % 25:
            print(epoch, _batch_mean_loss, val_r2)

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
    # d = load_data()
    # d, scalers = fit_transform(d)
    d, scalers = joblib.load('data_multi.pth')

    train_d = d
    loss_fn = torch.nn.MSELoss()
    dl = get_dataloader(train_d, batch_size)
    # train one epoch
    epoch_log_buffer = []
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for no, comp, proc_bool, proc_scalar, prop, mask, elem_t in dl:
            # forward pass
            out = model(comp, elem_t, proc_bool, proc_scalar)
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
        model, d, scalers=train_a_model(save_path=save_path, num_training_epochs=2000)
        torch.save(model.state_dict(), default_model_pth)
        joblib.dump((d, scalers), default_data_pth)
    
    return model, d, scalers

if __name__ == '__main__':
    get_model(f'model_multi_DNN.pth',f'data_multi.pth',resume=False,save_path='model_multi_DNN_train_err_log.txt')
    validate_a_model(num_training_epochs=2000, save_path='model_multi_valid_log_DNN.txt')
