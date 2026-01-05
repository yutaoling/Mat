

'''import matplotlib.pyplot as plt
import numpy as np

# 文件列表和标签
files_and_labels = {
    'log\\dar_train_err_log.txt': 'DAR',
    'log\\el_train_err_log.txt': 'EL',
    'log\\hv_train_err_log.txt': 'HV',
    'log\\uts_train_err_log.txt': 'UTS',
    'log\\ym_train_err_log.txt': 'YM',
    'log\\ys_train_err_log.txt': 'YS'
}

plt.figure(figsize=(12, 8))

for file, label in files_and_labels.items():
    try:
        data = np.loadtxt(file)
        plt.plot(data[:, 0], data[:, 1], label=label, linewidth=2)
    except FileNotFoundError:
        print(f"文件 {file} 未找到")

plt.xlabel('Epoch/Step')
plt.ylabel('Error/Loss')
plt.title('Training Error Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()'''

'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['Arial']# 'SimHei', 
plt.rcParams['axes.unicode_minus'] = False

# 文件名列表
file_names = ['model_multi_valid_log_DNN.txt', 'model_multi_valid_log_RNN.txt']

# 类别名称（用于图例）
categories = ['DNN', 'RNN']

# 创建2x3的子图布局
fig, axes = plt.subplots(3, 9, figsize=(12, 8))

axes = axes.flatten()  # 将二维数组展平为一维，便于遍历

# 颜色和线型设置
# 颜色和线型设置
colors = ['black', 'blue', 'red', 'green']  # 修改：提供三种颜色
line_styles = ['-', '-', '-', '-']  # 修改：提供三种线型
line_labels = ['Training Loss', 'Valid Loss', 'Scalar T L', 'Scalar V L']  # 修改：提供三种标签
# 遍历每个文件
n_c_list = [0, 1, 2]
n_l_list = [1, 2, 3]
n_n_list = [16, 32, 64]
for i in range(3):
    for j in range(3):
        for k in range(3):
            file_name = f'model_valid_log/1219_012_123_136_tuned_dataset_2/model_valid_log_{n_c_list[i]}_CNN_{n_l_list[j]}_DNN_{n_n_list[k]}_nerons.txt'

                # for i, (file_name, category) in enumerate(zip(file_names, categories)):
            try:
                # 读取数据
                data = np.loadtxt(file_name)

                # 提取各列数据
                epochs = data[:, 0]
                train_loss = data[:, 1]
                valid_loss = data[:, 2]

                # 在当前子图中绘制三条曲线
                ax = axes[9*k+3*i+j]
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.plot(epochs, train_loss, color=colors[0], linestyle=line_styles[0],
                        label=line_labels[0], linewidth=0.5)
                ax.plot(epochs, valid_loss, color=colors[1], linestyle=line_styles[1],
                        label=line_labels[1], linewidth=0.5)



                # 设置子图标题和标签
                ax.set_title(f'{n_c_list[i]} CNN {n_l_list[j]} DNN {n_n_list[k]} N', fontsize=10, fontweight='regular')
                # ax.set_xlabel('Epoch', fontsize=8)
                # ax.set_ylabel('Loss', fontsize=8)
                ax.legend(fontsize=6)
                ax.grid(True, alpha=0.3, which='both', axis='y')
                
                # plt.ylim(1e-2,1)

                # 设置坐标轴范围（可根据需要调整）
                ax.set_xlim(left=10, right=1e3)
                ax.set_ylim(bottom=0.1, top=1)
                ax.yaxis.set_major_formatter('{x:.1f}')
                ax.yaxis.set_minor_formatter('{x:.1f}')

                ax.tick_params(axis='both', which='both', labelsize=6)

            except FileNotFoundError:
                print(f"警告: 文件 {file_name} 未找到，跳过该文件")
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")

            file_name = f'model_valid_log/1219_012_123_136_tuned_dataset_scalar_2/model_valid_log_{n_c_list[i]}_CNN_{n_l_list[j]}_DNN_{n_n_list[k]}_nerons.txt'
'''
'''
    # for i, (file_name, category) in enumerate(zip(file_names, categories)):
            try:
                # 读取数据
                data = np.loadtxt(file_name)

                # 提取各列数据
                epochs = data[:, 0]
                train_loss = data[:, 1]
                valid_loss = data[:, 2]

                # 在当前子图中绘制三条曲线
                ax = axes[9*k+3*i+j]
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.plot(epochs, train_loss, color=colors[2], linestyle=line_styles[2],
                        label=line_labels[2], linewidth=0.5)
                ax.plot(epochs, valid_loss, color=colors[3], linestyle=line_styles[3],
                        label=line_labels[3], linewidth=0.5)



                # 设置子图标题和标签
                ax.set_title(f'{n_c_list[i]} CNN {n_l_list[j]} DNN {n_n_list[k]} N', fontsize=10, fontweight='regular')
                # ax.set_xlabel('Epoch', fontsize=8)
                # ax.set_ylabel('Loss', fontsize=8)
                ax.legend(fontsize=6)
                ax.grid(True, alpha=0.3, which='both', axis='y')
                
                # plt.ylim(1e-2,1)

                # 设置坐标轴范围（可根据需要调整）
                ax.set_xlim(left=10, right=1e3)
                ax.set_ylim(bottom=0.1, top=1)
                ax.yaxis.set_major_formatter('{x:.1f}')
                ax.yaxis.set_minor_formatter('{x:.1f}')

                ax.tick_params(axis='both', which='both', labelsize=6)

            except FileNotFoundError:
                print(f"警告: 文件 {file_name} 未找到，跳过该文件")
            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")
'''
'''
# 调整子图间距
plt.tight_layout(pad=0.5)

# 添加总标题
#plt.suptitle('Training and Validation Loss Comparison Across Categories',
#             fontsize=16, fontweight='bold', y=0.98)

# 保存图片（可选）
# plt.savefig('validation_loss_comparison.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['Arial']# 'SimHei', 
plt.rcParams['axes.unicode_minus'] = False

# 文件名列表
file_names = ['model_multi_valid_log_DNN.txt', 'model_multi_valid_log_RNN.txt']

# 类别名称（用于图例）
categories = ['DNN', 'RNN']

# 创建2x3的子图布局
fig, axes = plt.subplots(1, 1, figsize=(12, 8))

# axes = axes.flatten()  # 将二维数组展平为一维，便于遍历

# 颜色和线型设置
# 颜色和线型设置
colors = ['black', 'blue', 'red', 'green']  # 修改：提供三种颜色
line_styles = ['-', '-', '-', '-']  # 修改：提供三种线型
line_labels = ['Training Loss', 'Valid Loss', 'Scalar T L', 'Scalar V L']  # 修改：提供三种标签
# 遍历每个文件
file_name = f'model_multi_valid_log_DNN.txt'

    # for i, (file_name, category) in enumerate(zip(file_names, categories)):
try:
    # 读取数据
    data = np.loadtxt(file_name)

    # 提取各列数据
    epochs = data[:, 0]
    train_loss = data[:, 1]
    valid_loss = data[:, 2]

    # 在当前子图中绘制三条曲线
    ax = axes# [0]
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(epochs, train_loss, color=colors[0], linestyle=line_styles[0],
            label=line_labels[0], linewidth=1)
    ax.plot(epochs, valid_loss, color=colors[1], linestyle=line_styles[1],
            label=line_labels[1], linewidth=1)



    # 设置子图标题和标签
    ax.set_title(f'N', fontsize=14, fontweight='regular')
    # ax.set_xlabel('Epoch', fontsize=8)
    # ax.set_ylabel('Loss', fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both', axis='y')
    
    # plt.ylim(1e-2,1)

    # 设置坐标轴范围（可根据需要调整）
    ax.set_xlim(left=10, right=1e3)
    ax.set_ylim(bottom=0.01, top=1)
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.yaxis.set_minor_formatter('{x:.2f}')

    ax.tick_params(axis='both', which='both', labelsize=10)

except FileNotFoundError:
    print(f"警告: 文件 {file_name} 未找到，跳过该文件")
except Exception as e:
    print(f"处理文件 {file_name} 时出错: {e}")
'''
'''
# 调整子图间距
plt.tight_layout(pad=0.5)

# 添加总标题
#plt.suptitle('Training and Validation Loss Comparison Across Categories',
#             fontsize=16, fontweight='bold', y=0.98)

# 保存图片（可选）
# plt.savefig('validation_loss_comparison.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

'''
import matplotlib.pyplot as plt
import joblib
import numpy as np

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 定义pkl文件列表和对应的标签
pkl_files = [
    'rl_single_agent_direct_R-0b742507.pkl',
    'rl_single_agent_direct_R-4de95647.pkl',
    'rl_single_agent_direct_R-c398fc6b.pkl',
    'rl_single_agent_direct_R-c657aec7.pkl',
    'rl_single_agent_direct_R-ef7da07a.pkl'
]

# 对应的标签名称（您可以根据实际情况修改）
labels = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']

# 颜色设置
colors = ['blue', 'red', 'green', 'orange', 'purple']
line_styles = ['-', '--', '-.', ':', '-']

# 创建图表
plt.figure(figsize=(12, 8))

# 遍历每个pkl文件
for i, (pkl_file, label, color, line_style) in enumerate(zip(pkl_files, labels, colors, line_styles)):
    try:
        # 读取pkl文件
        bsf_list = joblib.load(pkl_file)

        # 生成epoch序列（从0开始）
        epochs = list(range(len(bsf_list)))

        # 绘制曲线
        plt.plot(epochs, bsf_list, color=color, linestyle=line_style,
                 label=label, linewidth=2, alpha=0.8)

        print(f"成功加载 {pkl_file}: 共 {len(bsf_list)} 个数据点")

    except FileNotFoundError:
        print(f"警告: 文件 {pkl_file} 未找到，跳过该文件")
    except Exception as e:
        print(f"处理文件 {pkl_file} 时出错: {e}")

# 设置图表属性
plt.title('Best Score Evolution Across Different Runs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Best Score', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 设置坐标轴范围
plt.xlim(left=0)
# 自动调整y轴范围以显示所有数据

# 添加一些统计信息（可选）
plt.tight_layout()

# 保存图片（可选）
# plt.savefig('rl_best_score_comparison.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
'''