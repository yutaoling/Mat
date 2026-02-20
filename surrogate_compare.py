import torch
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from surrogate_train import *

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compare(model, dataloader, results, index, scalers):
    model_name = model.get_name()
    print(f'\nModel: {model_name}')
    model_path = f'models/surrogate/model_{model_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    dl = get_dataloader(dataloader, batch_size=len(dataloader[0]), augment=False)
    prop_original = np.zeros((0, N_PROP))
    pred_original = np.zeros((0, N_PROP))
    prop_mask_np = None
    for batch in dl:
        id_t, comp_t, pb_t, ps_t, ph_t, prop_t, elem_t, pbm_t, psm_t, pm_t = batch
        with torch.no_grad():
            pred = model(comp_t, elem_t, pb_t, ps_t, ph_t, pbm_t, psm_t, scalers)

        pred_np = pred.detach().reshape(-1, N_PROP).cpu().numpy()
        prop_np = prop_t.detach().reshape(-1, N_PROP).cpu().numpy()
        prop_mask_np_batch = pm_t.detach().reshape(-1, N_PROP).cpu().numpy()

        prop_scaler = scalers[4]
        pred_ori = prop_scaler.inverse_transform(pred_np)
        prop_ori = prop_scaler.inverse_transform(prop_np)
        prop_ori[prop_ori < 0.1] = np.nan

        pred_original = np.concatenate((pred_original, pred_ori), axis=0)
        prop_original = np.concatenate((prop_original, prop_ori), axis=0)

        if prop_mask_np is None:
            prop_mask_np = prop_mask_np_batch
        else:
            prop_mask_np = np.concatenate((prop_mask_np, prop_mask_np_batch), axis=0)

    for i, prop_name in enumerate(PROP):
        valid_mask = prop_mask_np[:, i] == 1
        y_true = prop_original[valid_mask, i]
        y_pred = pred_original[valid_mask, i]
        
        valid_idx = ~np.isnan(y_true)
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
        
        if len(y_true) == 0:
            continue

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        print(f'On {prop_name}: R²={r2:.3f}\tMAE={mae:.1f}\tRMSE={rmse:.1f}\tMAPE={mape:.1f}%\tn={len(y_true)}')
        results.loc[index] = [model_name, prop_name, r2, mae, rmse, mape, len(y_true)]
        index += 1
    return results, index

def surrogate_compare():
    results = pd.DataFrame(columns=['Model', 'Target', 'R2', 'MAE', 'RMSE', 'MAPE', 'N'])
    index = 0
    
    train_d, val_d, scalers = joblib.load('models/surrogate/data.pth')
    mask_modes = ['zero', 'learned', 'mean_dropout', 'sample_dropout']
    for _type in ['val']:
        for mask_mode in mask_modes:
            for model in MODEL_LIST(mask_mode = mask_mode):
                results, index = compare(model, val_d, results, index, scalers)
    model = TiAlloyNet().to(device)
    results, index = compare(model, val_d, results, index, scalers)
    results.to_excel(f'results/surrogate_comparison.xlsx', index=False)
    
if __name__ == "__main__":
    surrogate_compare()