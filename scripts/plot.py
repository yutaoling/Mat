"""绘图工具模块"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate_train import *
import matplotlib.pyplot as plt
import numpy as np
import joblib


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_plot_style():
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300


def plot_training_errors():
    model_names = [
    'ELM_CPrPh', 'ELM_C', 'ELM_CPh', 'ELM_CPr', 
    'ELM_ElemFeat', 'ELM_CNN',
    'FCNN', 'FCNN_MSHBranched', 'FCNN_FullyBranched',
    'FCNN_ElemFeat', 'FCNN_ElemFeat_MSHBranched', 'FCNN_ElemFeat_FullyBranched',
    'CNN', 'CNN_MSHBranched', 'CNN_FullyBranched',
    'Attention', 'Attention_MSHBranched', 'Attention_FullyBranched',
    'TiAlloyNet',
]
    file_paths = [os.path.join(PROJECT_ROOT, f'logs/surrogate/train_{model_name}.txt') for model_name in model_names]
    labels = model_names
    setup_plot_style()
    plt.figure(figsize=(6, 4))
    colors = ['#2c3e50', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#27ae60', '#1abc9c', '#d35400', '#7f8c8d', '#8e44ad',
              '#2ecc71', '#e67e22', '#34495e', '#16a085', '#f1c40f', '#2980b9', '#808b96', '#2c3e50', '#3498db', '#e74c3c', '#9b59b6']
    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        try:
            data = np.loadtxt(file_path)
            # plt.plot(data[:, 0], data[:, 1], label=label+'_T', linewidth=1, color=colors[i], linestyle='-')
            plt.plot(data[:, 0], data[:, 2], label=label+'_V', linewidth=1, color=colors[i], linestyle='-')
        except Exception:
            pass
    
    plt.xlabel('Epoch/Step', fontsize=12)
    plt.ylabel('Error/Loss', fontsize=12)
    plt.ylim(bottom=0, top=1)
    plt.xlim(left=0)
    plt.title('Training Error Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    
    plt.show()


def plot_model_comparison():
    file_paths = [os.path.join(PROJECT_ROOT, f'logs/surrogate/log_{model_name}.txt') for model_name in MODEL_NAMES]
    labels = MODEL_NAMES
    setup_plot_style()
    
    n_models = len(file_paths)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    colors = ['#2c3e50', '#3498db']
    
    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        ax = axes[i]
        try:
            data = np.loadtxt(file_path)
            ax.plot(data[:, 0], data[:, 1], color=colors[0], label='Training', linewidth=1.5)
            ax.plot(data[:, 0], data[:, 2], color=colors[1], label='Validation', linewidth=1.5)
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.annotate(f'Final: {data[-1, 2]:.4f}', xy=(data[-1, 0], data[-1, 2]),
                       xytext=(-60, 20), textcoords='offset points', fontsize=9, color=colors[1],
                       arrowprops=dict(arrowstyle='->', color=colors[1], lw=0.5))
        except Exception:
            ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=14)
    
    plt.tight_layout()
    
    plt.show()


def plot_rl_best_scores():
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    data_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) 
        if (f.endswith('.pkl') or f.endswith('.txt')) and 'rl_single_agent' in f]
    
    setup_plot_style()
    
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(pkl_files))]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(pkl_files)))
    line_styles = ['-', '--', '-.', ':', '-']
    
    plt.figure(figsize=(12, 8))
    all_final_scores = []
    
    for i, (data_file, label) in enumerate(zip(pkl_files, labels)):
        try:
            if data_file.endswith('.pkl'):
                bsf_list = joblib.load(data_file)
            else:
                bsf_list = np.loadtxt(data_file)
            style = line_styles[i % len(line_styles)]
            plt.plot(range(len(bsf_list)), bsf_list, color=colors[i], linestyle=style,
                     label=f'{label} (final: {bsf_list[-1]:.4f})', linewidth=2, alpha=0.8)
            all_final_scores.append(bsf_list[-1])
        except Exception:
            pass
    
    plt.title('Best Score Evolution', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Best Score', fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    
    if all_final_scores:
        mean_score = np.mean(all_final_scores)
        plt.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    plt.show()


def plot_prediction_scatter(model=None):
    """绘制验证集上真实值-预测值对比的对角线图"""
    import torch
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    setup_plot_style()
    
    prop_names = PROP
    
    train_d, val_d, scalers = joblib.load('models/surrogate/data.pth')
    
    if model is None:
        model_path = os.path.join(PROJECT_ROOT, 'models/surrogate/model_ELM_CPrPh.pth')
        model = FCNN_NoElemFeat_Model()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    model = model.to(device)
    model.eval()
    
    val_dl = get_dataloader(val_d, batch_size=len(val_d[0]), augment=False)
    _, comp, proc_bool, proc_scalar, phase_scalar, prop, mask, elem_t = next(iter(val_dl))
    
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
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    units = ['GPa', 'MPa', 'MPa', '%', 'HV']
    
    for i, (prop_name, unit) in enumerate(zip(prop_names, units)):
        if i >= len(axes):
            break
        ax = axes[i]
        
        valid_mask = mask_np[:, i] == 1
        y_true = prop_original[valid_mask, i]
        y_pred = pred_original[valid_mask, i]
        
        valid_idx = ~np.isnan(y_true)
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
        
        if len(y_true) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{prop_name}', fontsize=12)
            continue
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, c='#3498db', edgecolors='white', linewidth=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.1
        line_range = [min_val - margin, max_val + margin]
        ax.plot(line_range, line_range, 'r--', linewidth=2, label='y=x')
        ax.fill_between(line_range, [x * 0.9 for x in line_range], [x * 1.1 for x in line_range],
                       alpha=0.15, color='green', label='±10%')
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        stats_text = f'R²={r2:.3f}\nMAE={mae:.1f}\nRMSE={rmse:.1f}\nMAPE={mape:.1f}%\nn={len(y_true)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Actual {prop_name} ({unit})', fontsize=10)
        ax.set_ylabel(f'Predicted {prop_name} ({unit})', fontsize=10)
        ax.set_title(f'{prop_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=6)
        ax.set_xlim(line_range)
        ax.set_ylim(line_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    for i in range(len(prop_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Validation Set: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    
    plt.show()
    
    return pred_original, prop_original, mask_np




if __name__ == '__main__':
    plot_training_errors()
    # plot_model_comparison()
    # plot_rl_best_scores()
    # plot_prediction_scatter(TiAlloyNet())
