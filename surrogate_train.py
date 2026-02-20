import torch
import os
import random
from typing import Callable, Tuple
import warnings
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, InconsistentVersionWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from surrogate_model import *


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

    proc_bool_mask = ~np.isnan(proc_bool_data)
    proc_scalar_mask = ~np.isnan(proc_scalar_data)
    prop_mask = ~np.isnan(prop_data)
    
    d = (id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature, proc_bool_mask, proc_scalar_mask, prop_mask,)

    return d

def fit_transform(data_tuple, scalers = None):
    id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature, proc_bool_mask, proc_scalar_mask, prop_mask = data_tuple
    
    def scale_data(data, mask, scaler=None):
        if scaler is None:
            scaler = StandardScaler()
            if mask is not None:
                valid_data = data[mask].reshape(-1, 1) if data.ndim == 1 else data[mask.any(axis=1)]
                scaler.fit(data)
            else:
                scaler.fit(data)
        scaled_data = data.copy()
        mean = scaler.mean_
        std = scaler.scale_
        scaled_data = (data - mean) / (std + 1e-8)
        scaled_data = np.nan_to_num(scaled_data, nan=0.0)
        return scaled_data, scaler

    if scalers is not None:
        comp_data = scalers[0].transform(comp_data)
        proc_bool_data = np.nan_to_num((proc_bool_data - scalers[1].mean_) / scalers[1].scale_, nan=0.0)
        proc_scalar_data = np.nan_to_num((proc_scalar_data - scalers[2].mean_) / scalers[2].scale_, nan=0.0)
        phase_scalar_data = np.nan_to_num((phase_scalar_data - scalers[3].mean_) / scalers[3].scale_, nan=0.0)
        prop_data = np.nan_to_num((prop_data - scalers[4].mean_) / scalers[4].scale_, nan=0.0)
        elem_feature = scalers[5].transform(elem_feature.T)
        
        return (
            (id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature, proc_bool_mask, proc_scalar_mask, prop_mask,),
            scalers,
        )

    else:
        comp_scaler = StandardScaler().fit(comp_data)
        comp_data = comp_scaler.transform(comp_data)
        
        def get_scaler_with_nan(data):
            s = StandardScaler()
            flat_data = data.flatten()
            valid_elements = flat_data[~np.isnan(flat_data)].reshape(-1, 1)
            if len(valid_elements) > 0:
                s.fit(valid_elements)
            else:
                s.mean_ = np.array([0.0])
                s.scale_ = np.array([1.0])
            return s
        
        def smart_scale(data):
            df = pd.DataFrame(data)
            scaler = StandardScaler()
            scaler.fit(df)
            scaled_values = scaler.transform(df)
            return np.nan_to_num(scaled_values, nan=0.0), scaler

        proc_bool_data, proc_bool_scaler = smart_scale(proc_bool_data)
        proc_scalar_data, proc_scalar_scaler = smart_scale(proc_scalar_data)
        phase_scalar_data, phase_scalar_scaler = smart_scale(phase_scalar_data)
        prop_data, prop_scaler = smart_scale(prop_data)
        
        elem_feature_scaler = StandardScaler()
        elem_feature = elem_feature_scaler.fit_transform(elem_feature.T)

        return (
            (id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, elem_feature, proc_bool_mask, proc_scalar_mask, prop_mask,),
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

        self.proc_bool_mask = self.data_tuple[7]
        self.proc_scalar_mask = self.data_tuple[8]
        self.prop_mask = self.data_tuple[9]
        
    def __len__(self):
        return len(self.comp)
    
    def __getitem__(self, idx):
        _id = self.id[idx]
        _comp = self.comp[idx].copy()
        _proc_bool = self.proc_bool[idx]
        _proc_scalar = self.proc_scalar[idx].copy()
        _phase_scalar = self.phase_scalar[idx].copy()
        _prop = self.prop[idx].copy()
        _proc_bool_mask = self.proc_bool_mask[idx]
        _proc_scalar_mask = self.proc_scalar_mask[idx]
        _prop_mask = self.prop_mask[idx]

        if self.augment:
            noise_comp = np.random.normal(0,self.noise_std,_comp.shape)
            _comp += noise_comp
            noise_proc = np.random.normal(0, self.noise_std, _proc_scalar.shape)
            _proc_scalar += noise_proc * _proc_scalar_mask
            noise_phase = np.random.normal(0, self.noise_std, _phase_scalar.shape)
            _phase_scalar += noise_phase

        return _id, _comp, _proc_bool, _proc_scalar, _phase_scalar, _prop, _proc_bool_mask, _proc_scalar_mask, _prop_mask

def get_dataloader(data_tuple, batch_size = 16, augment = False) -> DataLoader:
    id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, \
        elem_feature, proc_bool_mask, proc_scalar_mask, prop_mask = data_tuple
    dataset = CustomDataset(data_tuple, None, augment=augment)

    _elem_feature_base = torch.tensor(elem_feature, dtype=torch.float32, device=device).reshape(1, 1, *elem_feature.shape)

    def _collate_fn(batch):
        id, comp, proc_bool, proc_scalar, phase_scalar, prop, proc_bool_mask, proc_scalar_mask, prop_mask = zip(*batch)
        bs = len(comp)
        
        id_t = torch.tensor(np.vstack(id), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        comp_t = torch.tensor(np.vstack(comp), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        proc_bool_t = torch.tensor(np.vstack(proc_bool), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        proc_scalar_t = torch.tensor(np.vstack(proc_scalar), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        phase_scalar_t = torch.tensor(np.vstack(phase_scalar), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        prop_t = torch.tensor(np.vstack(prop), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        proc_bool_mask_t = torch.tensor(np.vstack(proc_bool_mask), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        proc_scalar_mask_t = torch.tensor(np.vstack(proc_scalar_mask), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        prop_mask_t = torch.tensor(np.vstack(prop_mask), dtype=torch.float32, device=device).reshape(bs, 1, -1, 1)
        
        elem_t = _elem_feature_base.expand(bs, 1, *elem_feature.shape)

        return id_t, comp_t, proc_bool_t, proc_scalar_t, phase_scalar_t, prop_t, elem_t, proc_bool_mask_t, proc_scalar_mask_t, prop_mask_t

    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn, shuffle=augment)

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
            # if self.verbose:
                # print(f"[EarlyStopping] New best val loss: {val_loss:.6f}")
            return False
        else:
            self.counter += 1
            # if self.verbose and self.counter % 25 == 0:
                # print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")
            return self.counter >= self.patience

_cached_loss_tensors = {}

def _get_cached_tensors(dtype, device, prop_scaler):
    key = (dtype, device)
    if key not in _cached_loss_tensors or _cached_loss_tensors[key].get('scaler_id') != id(prop_scaler):
        scaler_mean = torch.tensor(prop_scaler.mean_, dtype=dtype, device=device).view(1, -1)
        scaler_scale = torch.tensor(prop_scaler.scale_, dtype=dtype, device=device).view(1, -1)
        _cached_loss_tensors[key] = {
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale,
            'scaler_id': id(prop_scaler)
        }
    return _cached_loss_tensors[key]

def MaskedLoss(out, prop, mask, prop_scaler=None):
    cached = _get_cached_tensors(out.dtype, out.device, prop_scaler)
    scaler_mean = cached['scaler_mean']
    scaler_scale = cached['scaler_scale']

    weights = torch.ones(out.shape[1], device=out.device).view(1, -1)

    huber = nn.functional.huber_loss(out, prop, reduction='none', delta=1)
    per_label_loss = huber * mask * weights

    out_original = out * scaler_scale + scaler_mean
    constraint_positive = nn.functional.relu(-out_original) * mask * weights

    per_label_total = per_label_loss + 10.0 * constraint_positive

    denom_labels = (mask * weights).sum().clamp(min=1e-6)
    loss_main = per_label_total.sum() / denom_labels

    pair_mask = (mask[:, 1] * mask[:, 2]).clamp(min=0.0, max=1.0)
    constraint_uts_ys = nn.functional.relu(out_original[:, 1] - out_original[:, 2]) * pair_mask
    denom_pairs = pair_mask.sum().clamp(min=1e-6)
    loss_pair = constraint_uts_ys.sum() / denom_pairs

    total = loss_main + 10.0 * loss_pair

    per_sample_denom = (mask * weights).sum(dim=1).clamp(min=1e-6)
    per_sample = per_label_total.sum(dim=1) / per_sample_denom + 10.0 * constraint_uts_ys

    return total, per_sample


def train_validate_split(
    data_tuple,
    ratio_tuple=(0.9, 0.1),
):
    _random_seed = next(iter(seeds))

    id_data, comp_data, proc_bool_data, proc_scalar_data, phase_scalar_data, prop_data, \
        elem_feature, proc_bool_mask, proc_scalar_mask, prop_mask = data_tuple

    val_ratio = ratio_tuple[1] / sum(ratio_tuple)

    id_train, id_val, comp_train, comp_val, proc_bool_train, proc_bool_val, proc_scalar_train, proc_scalar_val, \
        phase_scalar_train, phase_scalar_val, prop_train, prop_val, \
        proc_bool_mask_train, proc_bool_mask_val, proc_scalar_mask_train, proc_scalar_mask_val, \
        prop_mask_train, prop_mask_val = \
        train_test_split(
            id_data,
            comp_data,
            proc_bool_data,
            proc_scalar_data,
            phase_scalar_data,
            prop_data,
            proc_bool_mask,
            proc_scalar_mask,
            prop_mask,
            test_size=val_ratio,
            random_state=_random_seed,
        )

    train_d = (
        id_train, comp_train, proc_bool_train, proc_scalar_train, phase_scalar_train, prop_train,
        elem_feature, proc_bool_mask_train, proc_scalar_mask_train, prop_mask_train
    )
    val_d = (
        id_val, comp_val, proc_bool_val, proc_scalar_val, phase_scalar_val, prop_val,
        elem_feature, proc_bool_mask_val, proc_scalar_mask_val, prop_mask_val
    )

    return train_d, val_d

def validate(model, val_dl, scalers=None, prop_scaler=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_llist = []
    all_out = []
    all_prop = []
    all_mask = []
    all_id = []
    for batch in val_dl:
        id, comp, proc_bool, proc_scalar, phase_scalar, prop, elem_feat, proc_bool_mask, proc_scalar_mask, prop_mask = batch
        with torch.no_grad():
            out = model(comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers)
        prop = prop.reshape(*(out.shape))
        prop_mask = prop_mask.reshape(*(out.shape))
        loss, llist = MaskedLoss(out, prop, prop_mask, prop_scaler)
        total_loss += loss.item() * len(prop)
        total_samples += len(prop)
        all_llist.append(llist)
        all_out.append(out)
        all_prop.append(prop)
        all_mask.append(prop_mask)
        all_id.append(id)
    
    val_loss = total_loss / total_samples
    all_llist = torch.cat(all_llist)
    all_out = torch.cat(all_out)
    all_prop = torch.cat(all_prop)
    all_mask = torch.cat(all_mask)
    all_id = torch.cat(all_id)
    
    max_sample = None
    min_sample = None
    
    if len(all_llist) > 0:
        max_idx = all_llist.argmax()
        max_sample = {
            'loss': all_llist[max_idx].item(),
            'predicted': all_out[max_idx].detach().cpu().numpy(),
            'actual': all_prop[max_idx].detach().cpu().numpy(),
            'mask': all_mask[max_idx].detach().cpu().numpy(),
            'id': all_id[max_idx].detach().cpu().numpy(),
        }
        
        min_idx = all_llist.argmin()
        min_sample = {
            'loss': all_llist[min_idx].item(),
            'predicted': all_out[min_idx].detach().cpu().numpy(),
            'actual': all_prop[min_idx].detach().cpu().numpy(),
            'mask': all_mask[min_idx].detach().cpu().numpy(),
            'id': all_id[min_idx].detach().cpu().numpy(),
        }
    
    return val_loss, max_sample, min_sample

def train_a_model(model = None,
                    train_d = None,
                    val_d = None,
                    scalers = None,
                    num_training_epochs = 1000,
                    batch_size = 16,
                    save_path = None,):
    train_dl = get_dataloader(train_d, batch_size)
    val_dl = get_dataloader(val_d, batch_size, augment=False)
    epoch_log_buffer = []
    early_stopper = EarlyStopping(patience=100,min_delta=1e-4,verbose=True)
    prop_scaler = scalers[4]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode='min', factor=0.5, patience=30
    )
    
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for id, comp, proc_bool, proc_scalar, phase_scalar, prop, elem_feat, proc_bool_mask, proc_scalar_mask, prop_mask in train_dl:
            out = model(comp, elem_feat, proc_bool, proc_scalar, phase_scalar, proc_bool_mask, proc_scalar_mask, scalers)
            loss, llist = MaskedLoss(out, prop.reshape(*(out.shape)), prop_mask.reshape(*(out.shape)), prop_scaler)

            model.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            model.optimizer.step()
            
            _batch_loss_buffer.append(loss.item())
        
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        val_loss, max_sample_val, min_sample_val = validate(model, val_dl, scalers, prop_scaler)
        
        scheduler.step(val_loss)
        epoch_log_buffer.append((epoch, _batch_mean_loss, val_loss))
        if not epoch % 25:
            print(epoch, _batch_mean_loss, val_loss)
        if not epoch % 100 and False:
            if max_sample_val is not None:
                predicted_inv = scalers[4].inverse_transform(max_sample_val['predicted'].reshape(1, -1))
                actual_inv = max_sample_val['actual'].copy().reshape(1, -1)
                actual_inv = scalers[4].inverse_transform(actual_inv)
                actual_inv[actual_inv < 0.1] = np.nan
                print(f"Max loss sample {int(max_sample_val['id'].flatten()[0])}, loss={max_sample_val['loss']:.6f}")
                print(f"Predicted: {predicted_inv.flatten()}")
                print(f"Actual: {actual_inv.flatten()}")
            if min_sample_val is not None:
                predicted_inv = scalers[4].inverse_transform(min_sample_val['predicted'].reshape(1, -1))
                actual_inv = min_sample_val['actual'].copy().reshape(1, -1)
                actual_inv = scalers[4].inverse_transform(actual_inv)
                actual_inv[actual_inv < 0.1] = np.nan
                print(f"Min loss sample {int(min_sample_val['id'].flatten()[0])}, loss={min_sample_val['loss']:.6f}")
                print(f"Predicted: {predicted_inv.flatten()}")
                print(f"Actual: {actual_inv.flatten()}")

        if epoch > 499:
            if early_stopper.step(val_loss, model) and val_loss < 1.0:
                print(f"[EarlyStopping] Stop at epoch {epoch}")
                break

    if save_path:
        np.savetxt(
            save_path,
            np.array(epoch_log_buffer),
            fmt = '%.6f',
            delimiter = '\t',
        )
    
    return model

def get_model(model = None,
              model_path = 'model.pth',
              data_path = 'data.pth',
              resume = False,
              train = True,
              save_path=None,):
    try:
        train_d, val_d, scalers = joblib.load(data_path)
    except:
        d = load_data()
        d = filter_activated_data(d, activated_value=1)
        d, scalers = fit_transform(d)
        train_d, val_d = train_validate_split(d, ratio_tuple=(0.9, 0.1))
        joblib.dump((train_d, val_d, scalers), data_path)

    if resume:
        model.load_state_dict(torch.load(model_path, map_location=device))

    if train:
        model=train_a_model(model = model,
            train_d = train_d,
            val_d = val_d,
            scalers = scalers,
            save_path=save_path,
            batch_size=16,
            num_training_epochs=1000)
        torch.save(model.state_dict(), model_path)
    
    return model, train_d, val_d, scalers

if __name__ == '__main__':
    
    mask_modes = ['zero', 'learned', 'mean_dropout', 'sample_dropout']
    for mask_mode in mask_modes:
        for model in MODEL_LIST(mask_mode = mask_mode):
            model_name = model.get_name()
            print(f"\nTraining model: {model_name}\n")
            model_dir = f'models/surrogate'
            log_dir = f'logs/surrogate'
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            
            get_model(model,
                f'{model_dir}/model_{model_name}.pth',
                f'{model_dir}/data.pth',
                resume=False,
                train=True,
                save_path=f'{log_dir}/train_{model_name}.txt')
    
    for connect_mode in ['jump', 'emb', 'sep']:
        model = TiAlloyNet(connect_mode=connect_mode).to(device)
        model_name = model.get_name()
        print(f"\nTraining model: {model_name}\n")
        model_dir = f'models/surrogate'
        log_dir = f'logs/surrogate'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        get_model(model,
            f'{model_dir}/model_{model_name}.pth',
            f'{model_dir}/data.pth',
            resume=False,
            train=True,
            save_path=f'{log_dir}/train_{model_name}.txt')