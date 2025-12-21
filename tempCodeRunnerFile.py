file_name = f'model_valid_log/1219_012_123_136_tuned_dataset_scalar_2/model_valid_log_{n_c_list[i]}_CNN_{n_l_list[j]}_DNN_{n_n_list[k]}_nerons.txt'

    # for i, (file_name, category) in enumerate(zip(file_names, categories)):
            try:
                # 读取数据
                data = np.loadtxt(file_name)

                # 提取各列数据
                epochs = data[:, 0]
                train_loss = data[:, 1]
                valid_loss = data[:, 2]

                # 在当前子图中绘制三条曲线
                ax = axes[9*i+3*j+k]
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