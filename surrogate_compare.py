import torch
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from surrogate_train import *

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def surrogate_compare():
    prop_names = PROP

    results = pd.DataFrame(columns=['Model', 'Target', 'R2', 'MAE', 'RMSE', 'MAPE', 'N'])
    index = 0
    
    train_d, val_d, scalers = joblib.load('models/surrogate/data.pth')
    val_dl = get_dataloader(val_d, batch_size=len(val_d[0]), augment=False)
    _, comp, proc_bool, proc_scalar, phase_scalar, prop, mask, elem_t = next(iter(val_dl))
    
    for model, model_name in zip(MODEL_LIST, MODEL_NAMES):
        print(f'\nModel: {model_name}')
        model_path = f'models/surrogate/model_{model_name}.pth'
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    
        with torch.no_grad():
            pred = model(comp, elem_t, proc_bool, proc_scalar, phase_scalar)
        
        pred_np = pred.cpu().numpy()
        prop_np = prop.reshape(pred_np.shape).cpu().numpy()
        mask_np = mask.reshape(pred_np.shape).cpu().numpy()
        
        prop_scaler = scalers[4]
        pred_original = prop_scaler.inverse_transform(pred_np)
        
        prop_np_copy = prop_np.copy()
        prop_np_copy[prop_np_copy == -1] = np.nan
        prop_original = prop_scaler.inverse_transform(prop_np_copy)

        for i, prop_name in enumerate(prop_names):
            valid_mask = mask_np[:, i] == 1
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
    results.to_excel('results/surrogate_comparison.xlsx', index=False)
    
if __name__ == "__main__":
    surrogate_compare()