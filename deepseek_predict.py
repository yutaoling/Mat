from model_env_train import *


def predict_all_properties(composition_input, proc_input=None):
    """
    快速预测材料的所有六种性质

    参数:
        composition_input: 元素组成列表，长度20，对应以下元素顺序：
            ['C(at%)', 'N(at%)', 'O(at%)', 'Al(at%)', 'Si(at%)', 'Sc(at%)',
             'Ti(at%)', 'V(at%)', 'Cr(at%)', 'Mn(at%)', 'Fe(at%)',
             'Ni(at%)', 'Cu(at%)', 'Zr(at%)', 'Nb(at%)', 'Mo(at%)', 'Sn(at%)',
             'Hf(at%)', 'Ta(at%)', 'W(at%)']
        proc_input: 处理条件，默认为None时使用[0]（标准化后的0值）

    返回:
        dict: 包含六种性质预测结果的字典
    """
    if proc_input is None:
        proc_input = [0]  # 标准化后的0值对应原始数据的平均值

    # 确保输入格式正确
    if len(composition_input) != 20:
        raise ValueError(f"元素组成输入长度应为20，当前为{len(composition_input)}")

    comp_array = np.array([composition_input], dtype=np.float32)
    proc_array = np.array([proc_input], dtype=np.float32)

    predictions = {}

    try:
        # 加载对应性质的模型和数据
        model_path = f'model_multi.pth'
        data_path = f'data_multi.pth'

        # 加载模型
        model = CnnDnnModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 加载数据和scaler
        (d, scalers) = joblib.load(data_path)
        comp_data_scaler, proc_data_scaler, prop_data_scaler, elem_feature_scaler = scalers

        # 获取元素特征
        elem_feature = d[3]  # 已经是标准化后的元素特征

        # 对输入进行标准化
        comp_scaled = comp_data_scaler.transform(comp_array)
        proc_scaled = proc_data_scaler.transform(proc_array)

        # 转换为tensor
        comp_tensor = torch.tensor(comp_scaled, dtype=torch.float32).reshape(-1, 1, comp_scaled.shape[-1], 1).to(
            device)
        proc_tensor = torch.tensor(proc_scaled, dtype=torch.float32).reshape(-1, 1, proc_scaled.shape[-1], 1).to(
            device)
        elem_tensor = torch.tensor(elem_feature, dtype=torch.float32).reshape(1, 1, *elem_feature.shape).to(device)

        # 预测
        with torch.no_grad():
            pred_scaled = model(comp_tensor, elem_tensor, proc_tensor)
            pred = prop_data_scaler.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, N_PROP))

        predictions = pred[0]
        # predictions[prop] = float(pred[0, 0])

    except Exception as e:
        print(f"预测性质时出错: {e}")
        predictions = 0
        # predictions[prop] = None

    return predictions


def predict_from_string(composition_str):
    """
    从字符串输入预测材料性质

    参数:
        composition_str: 空格分隔的元素组成字符串
        例如: "0.0001 0.0008 0.0188 0.1257 0.0188 0.0001 0.2521 0.0035 0.0061 0.0061 0.0284 0.0008 0.016 0.0008 0.0514 0.2992 0.0013 0.1516 0.0171 0.0013"

    返回:
        dict: 包含六种性质预测结果的字典
    """
    # 解析字符串输入
    comp_list = [float(x) for x in composition_str.strip().split()]
    return predict_all_properties(comp_list)


# 使用示例
if __name__ == '__main__':
    # 示例输入
    example_input = "0.0 0.0015 0.0015 0.0015 0.0102 0.0014 0.7086 0.0021 0.0021 0.0113 0.2475 0.0 0.0026 0.0026 0.0061 0.001 0.0 0.0 0.0 0.0"

    print("正在预测材料性质...")
    results = predict_from_string(example_input)

    print("\n预测结果:")
    print("=" * 50)
    print(example_input)
    for i, value in enumerate(results):
        if value is not None:
            label = PROP_LABELS[PROP[i]]
            print(f"{label}: {value:.4f}")
        else:
            print(f"{PROP_LABELS[PROP[i]]}: 预测失败")