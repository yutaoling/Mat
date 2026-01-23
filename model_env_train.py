import random
from typing import Callable, Tuple
import warnings
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, InconsistentVersionWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from model_env import ID, COMP, PROC_BOOL, PROC_SCALAR, PHASE_SCALAR, PROP
from model_env import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC_BOOL, N_PROC_SCALAR, N_PHASE_SCALAR, N_PROP
from model_env import CnnDnnModel, CNN_FCNN_MESH_Model, FCNN_Model, Attention_Model, device
from model_env import N_YM, N_YS, N_UTS, N_EL, N_HV, N_PROP_SAMPLE, N_SUM_PROP_SAMPLE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 0
set_seed(seed)
seeds = np.random.randint(0, 9999, (9999, ))


def load_data():
    data = pd.read_excel('data/Ti_Alloy_Dataset.xlsx', sheet_name=0)
    # data = data[data['Activated'] == 1]

    id_labels = ID
    comp_labels = COMP
    proc_bool_labels=PROC_BOOL
    proc_scalar_labels=PROC_SCALAR
    phase_scalar_labels = PHASE_SCALAR
    prop_labels = PROP
    cols_to_numeric = comp_labels + proc_bool_labels + proc_scalar_labels + phase_scalar_labels + prop_labels
    cols_to_numeric = [c for c in cols_to_numeric if c in data.columns]
    if cols_to_numeric:
        data[cols_to_numeric] = data[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

    id_data = data[id_labels].to_numpy()
    comp_data = data[comp_labels].to_numpy()
    proc_bool_data = data[proc_bool_labels].to_numpy()
    proc_scalar_data = data[proc_scalar_labels].to_numpy()
    phase_scalar_data = data[phase_scalar_labels].to_numpy()
    prop_data = data[prop_labels].to_numpy()

    elem_feature = pd.read_excel('data/elemental_features.xlsx')
    elem_feature = elem_feature[comp_labels].to_numpy()
    d = (id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature,)

    return d

def fit_transform(data_tuple, scalers = None):
    if scalers is not None:
        id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature = data_tuple
        comp_data = scalers[0].transform(comp_data)
        proc_bool_data = scalers[1].transform(proc_bool_data)
        proc_scalar_data = scalers[2].transform(proc_scalar_data)
        phase_scalar_data = scalers[3].transform(phase_scalar_data)
        prop_data = scalers[4].transform(prop_data)
        elem_feature = scalers[5].transform(elem_feature.T)
        return (
            (id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature,),
            scalers,
        )

    else:
        id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature = data_tuple
        comp_scaler, proc_bool_scaler, proc_scalar_scaler, phase_scalar_scaler, prop_scaler, elem_feature_scaler = \
            [StandardScaler() for _ in range(len(data_tuple)-1)]
        
        comp_data = comp_scaler.fit_transform(comp_data)
        proc_bool_data = proc_bool_scaler.fit_transform(proc_bool_data)
        proc_scalar_data = proc_scalar_scaler.fit_transform(proc_scalar_data)
        phase_scalar_data = phase_scalar_scaler.fit_transform(phase_scalar_data)
        prop_data = prop_scaler.fit_transform(prop_data)
        elem_feature = elem_feature_scaler.fit_transform(elem_feature.T)

        return (
            (id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature,),
            (comp_scaler, proc_bool_scaler, proc_scalar_scaler, phase_scalar_scaler, prop_scaler, elem_feature_scaler,),
        )

def filter_activated_data(data_tuple, activated_value=1):
    id_data = data_tuple[0]
    mask = id_data[:, 1] == activated_value

    filtered = []
    for i, item in enumerate(data_tuple):
        if i == 6:
            filtered.append(item)
        else:
            filtered.append(item[mask])

    return tuple(filtered)


class CustomDataset(Dataset):
    def __init__(self, 
                 data_tuple,
                 scaler: TransformerMixin,
                 augment: bool = False,
                 noise_std: float = 0.02):
        self.data_tuple = data_tuple
        self.scaler = scaler
        self.augment = augment
        self.noise_std = noise_std

        self.id = self.data_tuple[0]
        self.comp = self.data_tuple[1]
        self.proc_bool = self.data_tuple[2]
        self.proc_scalar = self.data_tuple[3]
        self.phase_scalar = self.data_tuple[4]
        self.prop = self.data_tuple[5]

        self.mask = ~np.isnan(self.prop)
        
    def __len__(self):
        return len(self.comp)
    
    def __getitem__(self, idx):
        _id = self.id[idx]
        _comp = self.comp[idx].copy()
        _proc_bool = self.proc_bool[idx]
        _proc_scalar = self.proc_scalar[idx].copy()
        _phase_scalar = self.phase_scalar[idx].copy()
        _prop = self.prop[idx]
        _mask = self.mask[idx]

        if self.augment:
            noise_comp = np.random.normal(0,self.noise_std,_comp.shape)
            _comp += noise_comp
            noise_proc = np.random.normal(0, self.noise_std, _proc_scalar.shape)
            _proc_scalar += noise_proc
            noise_phase = np.random.normal(0, self.noise_std, _phase_scalar.shape)
            _phase_scalar += noise_phase

        _prop = np.nan_to_num(_prop, nan=-1)
        
        return _id, _comp, _proc_bool, _proc_scalar, _phase_scalar, _prop, _mask

def get_dataloader(data_tuple, batch_size = 16, augment = False) -> DataLoader:
    id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature = data_tuple
    dataset = CustomDataset((id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data,), None, augment=augment)

    _elem_feature_tensor = torch.tensor(elem_feature, dtype=torch.float32).reshape(1, 1, *(elem_feature.shape))

    def _collate_fn(batch):
        id, comp, proc_bool, proc_scalar, phase_scalar, prop, mask = zip(*batch)
        id = torch.tensor(np.vstack(id), dtype=torch.float32)
        id = id.reshape(-1, 1, id.shape[-1], 1)
        comp = torch.tensor(np.vstack(comp), dtype=torch.float32)
        comp = comp.reshape(-1, 1, comp.shape[-1], 1)
        proc_bool = torch.tensor(np.vstack(proc_bool), dtype=torch.float32)
        proc_bool = proc_bool.reshape(-1, 1, proc_bool.shape[-1], 1)
        proc_scalar = torch.tensor(np.vstack(proc_scalar), dtype=torch.float32)
        proc_scalar = proc_scalar.reshape(-1, 1, proc_scalar.shape[-1], 1)
        phase_scalar = torch.tensor(np.vstack(phase_scalar), dtype=torch.float32)
        phase_scalar = phase_scalar.reshape(-1, 1, phase_scalar.shape[-1], 1)
        prop = torch.tensor(np.vstack(prop), dtype=torch.float32)
        prop = prop.reshape(-1, 1, prop.shape[-1], 1)
        mask = torch.tensor(np.vstack(mask), dtype=torch.float32)
        mask = mask.reshape(-1, 1, prop.shape[-1], 1)

        _elem_feature_tensor_clone = _elem_feature_tensor.expand(len(comp), 1, *(elem_feature.shape)).clone().detach()
        _elem_feature_tensor_clone.requires_grad_(False)

        id=id.to(device)
        comp=comp.to(device)
        proc_bool=proc_bool.to(device)
        proc_scalar=proc_scalar.to(device)
        phase_scalar=phase_scalar.to(device)
        prop=prop.to(device)
        mask=mask.to(device)
        _elem_feature_tensor_clone=_elem_feature_tensor_clone.to(device)

        return id, comp, proc_bool, proc_scalar, phase_scalar, prop, mask, _elem_feature_tensor_clone

    should_shuffle = augment

    return DataLoader(dataset, batch_size = batch_size, collate_fn = _collate_fn, shuffle = should_shuffle)

class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] New best val loss: {val_loss:.6f}")
            return False
        else:
            self.counter += 1
            if self.verbose and self.counter % 25 == 0:
                print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")
            return self.counter >= self.patience

def MaskedLoss(out, prop, mask, prop_scaler=None):
    """
    计算带物理约束的masked损失
    
    Args:
        out: 模型预测值（标准化后的）
        prop: 真实值（标准化后的）
        mask: 掩码，指示哪些属性有效
        prop_scaler: StandardScaler对象，用于反标准化到原始尺度计算物理约束
    """
    weights = torch.tensor(
        [1,1,1,1,1],# N_PROP_SAMPLE,
        dtype=out.dtype,
        device=out.device
    )
    weights = 1.0/weights
    weights = weights / weights.mean()
    weights = weights.view(1, -1)
    
    # 计算Huber损失（在标准化后的空间）
    loss = nn.functional.huber_loss(out, prop, reduction='none')
    loss = loss * mask
    loss = loss * weights
    

    prop_scaler_mean = torch.tensor(
        prop_scaler.mean_, 
        dtype=out.dtype, 
        device=out.device
    ).view(1, -1)
    prop_scaler_scale = torch.tensor(
        prop_scaler.scale_, 
        dtype=out.dtype, 
        device=out.device
    ).view(1, -1)
    
    out_original = out * prop_scaler_scale + prop_scaler_mean
    
    constraint_positive = nn.functional.relu(-out_original) * mask * weights
    constraint_uts_ys_masked = nn.functional.relu(out_original[:, 1] - out_original[:, 2]) * (mask[:, 1] * mask[:, 2])

    
    total_loss = loss + 10.0 * constraint_positive
    
    num_valid = (mask * weights).sum(dim=1).clamp(min=1e-6)
    mean_weighted_masked_loss = total_loss.sum(dim=1) / num_valid + 10.0 * constraint_uts_ys_masked
    
    return mean_weighted_masked_loss.mean(), mean_weighted_masked_loss

def train_validate_split(data_tuple, ratio_tuple = (0.95, 0.05)):
    _random_seed = next(iter(seeds))
    id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature = data_tuple
    _ratio_1 = sum(ratio_tuple[1:]) / sum(ratio_tuple)
    id_train, id_val, comp_train, comp_val, proc_bool_train, proc_bool_val, proc_scalar_train, proc_scalar_val, phase_scalar_train, phase_scalar_val, prop_train, prop_val = \
        train_test_split(id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, test_size = _ratio_1, random_state = _random_seed)
    return (id_train, comp_train, proc_bool_train, proc_scalar_train, phase_scalar_train, prop_train, elem_feature,), \
            (id_val, comp_val, proc_bool_val, proc_scalar_val, phase_scalar_val, prop_val, elem_feature,)

def validate(model, val_dl, prop_scaler=None, return_sample_info=False):
    model.to(device)
    model.eval()
    id, comp, proc_bool, proc_scalar, phase_scalar, prop, mask, elem_t = next(iter(val_dl))

    id = id.to(device)
    comp = comp.to(device)
    proc_bool = proc_bool.to(device)
    proc_scalar = proc_scalar.to(device)
    phase_scalar = phase_scalar.to(device)
    prop = prop.to(device)
    mask = mask.to(device)
    elem_t = elem_t.to(device)
    out = model(comp, elem_t, proc_bool, proc_scalar, phase_scalar).detach()
    prop = prop.reshape(*(out.shape))
    mask = mask.reshape(*(out.shape))
    loss, llist = MaskedLoss(out, prop, mask, prop_scaler)
    
    if not return_sample_info:
        return float(loss.item())
    
    # 找出验证集中loss最大和最小的样本
    max_sample = None
    min_sample = None
    
    max_loss = llist.max()
    max_idx = llist.argmax()
    max_sample = {
        'loss': max_loss.item(),
        'predicted': out[max_idx].detach().cpu().numpy(),
        'actual': prop[max_idx].detach().cpu().numpy(),
        'mask': mask[max_idx].detach().cpu().numpy(),
        'id': id[max_idx].detach().cpu().numpy(),
    }
    
    min_loss = llist.min()
    min_idx = llist.argmin()
    min_sample = {
        'loss': min_loss.item(),
        'predicted': out[min_idx].detach().cpu().numpy(),
        'actual': prop[min_idx].detach().cpu().numpy(),
        'mask': mask[min_idx].detach().cpu().numpy(),
        'id': id[min_idx].detach().cpu().numpy(),
    }
    
    return float(loss.item()), max_sample, min_sample

def Make_Masked_Data(train_data, test_data, rng_seed = seed):
    train_prop = train_data[5]
    test_prop = test_data[5]
    mask_prop = (~np.isnan(train_prop)).mean(axis=0)
    num_samples = test_prop.shape[0]
    num_props = mask_prop.shape[0]
    rng = np.random.default_rng(rng_seed)
    mask = rng.random((num_samples, num_props)) < mask_prop
    random_test_mask = None
    for _ in range(10):# max retry times
        valid = mask.sum(axis=1)>0
        if valid.all():
            random_test_mask = mask.astype(np.float32)
            break
        idx = np.where(~valid)[0]
        mask[idx]=rng.random((len(idx), num_props)) < mask_prop
    if random_test_mask is None:
        fallback_idx = np.argmax(mask_prop)
        random_test_mask = mask.astype(np.float32)
        invalid_rows = random_test_mask.sum(axis=1) == 0
        random_test_mask[invalid_rows, fallback_idx] = 1.0
    test_prop_masked = test_prop.copy()
    test_prop_masked[random_test_mask==0]=np.nan
    masked_test_data = (test_data[0], test_data[1], test_data[2], test_data[3], test_data[4], test_prop_masked, test_data[6])
    return masked_test_data


def validate_a_model(model = Attention_Model(),
                     num_training_epochs = 2000,
                     batch_size = 16,
                     save_path = None,
                     temp = None,):
    model = model.to(device) 
    d = load_data()
    d, scalers = fit_transform(d)
    d = filter_activated_data(d, activated_value=1)
    # test_data, scalers = fit_transform(test_data, scalers=scalers)
    # d, scalers = joblib.load('data_multi.pth')
    train_d, val_d = train_validate_split(d, (0.9, 0.1))
    train_dl = get_dataloader(train_d, batch_size, augment=True)
    val_dl = get_dataloader(val_d, batch_size, augment=False)
    epoch_log_buffer = []
    # early_stopper = EarlyStopping(patience=150,min_delta=1e-4,verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode='min', factor=0.5, patience=30
    )
    
    prop_scaler = scalers[4]
    
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for id, comp, proc_bool, proc_scalar, phase_scalar, prop, mask, elem_t in train_dl:
            id = id.to(device)
            comp = comp.to(device)
            proc_bool = proc_bool.to(device)
            proc_scalar = proc_scalar.to(device)
            phase_scalar = phase_scalar.to(device)
            prop = prop.to(device)
            elem_t = elem_t.to(device)
            out = model(comp, elem_t, proc_bool, proc_scalar, phase_scalar)
            l, _ = MaskedLoss(out, prop.reshape(*(out.shape)), mask.reshape(*(out.shape)), prop_scaler)

            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            _batch_loss_buffer.append(l.item())
        
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        
        # 在验证集上计算loss并获取最大/最小loss样本
        if not epoch % 25:
            val_loss, max_sample_val, min_sample_val = validate(model, val_dl, prop_scaler, return_sample_info=True)
        else:
            val_loss = validate(model, val_dl, prop_scaler, return_sample_info=False)
            
        scheduler.step(val_loss)
        epoch_log_buffer.append((epoch, _batch_mean_loss, val_loss))
        if not epoch % 25:
            print(epoch, _batch_mean_loss, val_loss)
            if max_sample_val is not None:
                predicted_inv = scalers[4].inverse_transform(max_sample_val['predicted'].reshape(1, -1))
                actual_inv = max_sample_val['actual'].copy().reshape(1, -1)
                actual_inv[actual_inv == -1] = np.nan
                actual_inv = scalers[4].inverse_transform(actual_inv)
                print(f"Max loss sample in val set (epoch {epoch}): Number={int(max_sample_val['id'].flatten()[0])}, loss={max_sample_val['loss']:.6f}")
                print(f"Predicted (original): {predicted_inv.flatten()}")
                print(f"Actual (original): {actual_inv.flatten()}")
            if min_sample_val is not None:
                predicted_inv = scalers[4].inverse_transform(min_sample_val['predicted'].reshape(1, -1))
                actual_inv = min_sample_val['actual'].copy().reshape(1, -1)
                actual_inv[actual_inv == -1] = np.nan
                actual_inv = scalers[4].inverse_transform(actual_inv)
                print(f"Min loss sample in val set (epoch {epoch}): Number={int(min_sample_val['id'].flatten()[0])}, loss={min_sample_val['loss']:.6f}")
                print(f"Predicted (original): {predicted_inv.flatten()}")
                print(f"Actual (original): {actual_inv.flatten()}")
        # if early_stopper.step(val_loss, model):
        #     print(f"[EarlyStopping] Stop at epoch {epoch}")
        #     break

    if save_path:
        np.savetxt(
            save_path,
            np.array(epoch_log_buffer),
            fmt = '%.6f',
            delimiter = '\t',
        )
    
    return model, d, scalers

def train_a_model(model = Attention_Model(),
                    num_training_epochs = 1000,
                    batch_size = 16,
                    save_path = None,):
    ''' train a model '''
    model = model.to(device)
    d = load_data()
    d, scalers = fit_transform(d)
    d = filter_activated_data(d, activated_value=1)
    # d, scalers = joblib.load('data_multi.pth')

    train_d = d
    train_dl = get_dataloader(train_d, batch_size)
    epoch_log_buffer = []
    
    prop_scaler = scalers[4]
    
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for id, comp, proc_bool, proc_scalar, phase_scalar, prop, mask, elem_t in train_dl:
            out = model(comp, elem_t, proc_bool, proc_scalar, phase_scalar)
            l, _ = MaskedLoss(out, prop.reshape(*(out.shape)), mask.reshape(*(out.shape)), prop_scaler)

            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            _batch_loss_buffer.append(l.item())
        
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

def get_model(model = Attention_Model(),
              default_model_pth = 'model.pth',
              default_data_pth = 'data.pth',
              resume = False,
              save_path=None,):

    if resume:
        model = model.to(device)
        model.load_state_dict(torch.load(default_model_pth, map_location=device))
        # 抑制 sklearn 版本不匹配警告（如果功能正常，可以忽略）
        with warnings.catch_warnings():
            # 过滤 sklearn InconsistentVersionWarning
            warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
            d, scalers = joblib.load(default_data_pth)
    else:
        model, d, scalers=train_a_model(model = model, save_path=save_path, num_training_epochs=2000)
        torch.save(model.state_dict(), default_model_pth)
        joblib.dump((d, scalers), default_data_pth)
    
    return model, d, scalers

if __name__ == '__main__':
    model = FCNN_Model()
    # get_model(model, f'models/surrogate/model_multi.pth',f'models/surrogate/data_multi.pth',resume=False,save_path='logs/surrogate/model_multi_train.txt')
    '''
    for n_c in [1, 2]:
        for n_n in [32, 64]:
            for n_l in [1, 2, 3]:
                print(f'Training model with {n_c} CNN layer(s), {n_l} DNN layer(s), {n_n} neurons per DNN layer')
                validate_a_model(num_training_epochs=1000, save_path=f'logs/validation/model_valid_log_{n_c}_CNN_{n_l}_DNN_{n_n}_nerons.txt', temp=[n_c,n_l,n_n])
    '''
    # validate_a_model(model = model, num_training_epochs=1000, batch_size=32, save_path='logs/surrogate/log_FCNN.txt')
    model = Attention_Model()
    get_model(model, f'models/surrogate/model_Attention.pth',f'models/surrogate/data_Attention.pth',resume=False,save_path='logs/surrogate/model_Attention_train.txt')
    
    # validate_a_model(model = model, num_training_epochs=1000, batch_size=32, save_path='logs/surrogate/log_Attention.txt')
