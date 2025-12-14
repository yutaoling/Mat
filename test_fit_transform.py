import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from model_env_train import fit_transform


def test_fit_transform_basic_functionality():
    """测试基本功能：标准化处理所有数据类型"""
    # 准备测试数据
    comp_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    proc_data = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    prop_data = np.array([[100.0], [200.0], [300.0]])
    elem_feature = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # (num_elem_features, num_elements)
    
    data_tuple = (comp_data, proc_data, prop_data, elem_feature)
    
    # 执行函数
    transformed_data, scalers = fit_transform(data_tuple)
    
    # 验证返回值结构
    assert len(transformed_data) == 4
    assert len(scalers) == 4
    
    # 验证每个scaler类型
    for scaler in scalers:
        assert isinstance(scaler, StandardScaler)
    
    # 验证数据形状
    comp_transformed, proc_transformed, prop_transformed, elem_feature_transformed = transformed_data
    assert comp_transformed.shape == comp_data.shape
    assert proc_transformed.shape == proc_data.shape
    assert prop_transformed.shape == prop_data.shape
    assert elem_feature_transformed.shape == (elem_feature.shape[1], elem_feature.shape[0])  # 转置


def test_fit_transform_single_sample():
    """测试单样本输入"""
    comp_data = np.array([[1.0, 2.0]])
    proc_data = np.array([[10.0, 20.0]])
    prop_data = np.array([[100.0]])
    elem_feature = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    
    data_tuple = (comp_data, proc_data, prop_data, elem_feature)
    transformed_data, scalers = fit_transform(data_tuple)
    
    # 单样本的标准差可能为0，但函数应该能正常处理
    assert transformed_data[0].shape == comp_data.shape


def test_fit_transform_boundary_values():
    """测试边界值：全零、全一、负值等"""
    # 全零数据
    comp_data = np.zeros((3, 2))
    proc_data = np.zeros((3, 3))
    prop_data = np.zeros((3, 1))
    elem_feature = np.zeros((2, 3))
    
    data_tuple = (comp_data, proc_data, prop_data, elem_feature)
    transformed_data, scalers = fit_transform(data_tuple)
    
    # 验证全零数据标准化后仍为数值（可能为NaN或inf，取决于scaler实现）
    assert not np.isnan(transformed_data[0]).any()
    
    # 全一数据
    comp_data = np.ones((2, 2))
    proc_data = np.ones((2, 2))
    prop_data = np.ones((2, 1))
    elem_feature = np.ones((3, 2))
    
    data_tuple = (comp_data, proc_data, prop_data, elem_feature)
    transformed_data, scalers = fit_transform(data_tuple)
    
    # 标准差为0的情况应该被正确处理
    assert transformed_data[0].shape == comp_data.shape


def test_fit_transform_element_feature_transpose():
    """测试元素特征的正确转置处理"""
    # 创建特定的元素特征数据
    elem_feature = np.array([
        [1.0, 2.0, 3.0],  # 特征1：Ti, Al, Zr的值
        [4.0, 5.0, 6.0],  # 特征2：Ti, Al, Zr的值
    ])  # 形状：(2, 3) - (num_elem_features, num_elements)
    
    comp_data = np.array([[0.5, 0.3, 0.2]])
    proc_data = np.array([[0.1, 0.9]])
    prop_data = np.array([[150.0]])
    
    data_tuple = (comp_data, proc_data, prop_data, elem_feature)
    transformed_data, scalers = fit_transform(data_tuple)
    
    elem_feature_transformed = transformed_data[3]
    
    # 验证转置：输入(2,3) -> 输出(3,2)
    assert elem_feature_transformed.shape == (3, 2)


def test_fit_transform_variable_lengths():
    """测试不同维度的数据"""
    # 不同样本数量
    comp_data = np.random.rand(10, 5)
    proc_data = np.random.rand(10, 8)
    prop_data = np.random.rand(10, 3)
    elem_feature = np.random.rand(4, 5)  # 有5个元素，4个特征
    
    data_tuple = (comp_data, proc_data, prop_data, elem_feature)
    transformed_data, scalers = fit_transform(data_tuple)
    
    # 验证形状一致性
    assert len(transformed_data[0]) == 10  # 样本数不变
    assert transformed_data[3].shape == (5, 4)  # 元素特征转置


def test_fit_transform_scaler_reusability():
    """测试返回的scaler是否可重用"""
    comp_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    proc_data = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    prop_data = np.array([[100.0], [200.0], [300.0]])
    elem_feature = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    data_tuple = (comp_data, proc_data, prop_data, elem_feature)
    transformed_data, scalers = fit_transform(data_tuple)
    
    # 使用返回的scaler对新数据进行转换
    new_comp_data = np.array([[2.0, 3.0]])
    transformed_new = scalers[0].transform(new_comp_data)
    
    assert transformed_new.shape == new_comp_data.shape


def test_fit_transform_standardization_effect():
    """验证标准化是否真正起作用"""
    # 创建有明显均值和方差的数据
    comp_data = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    
    # 只测试其中一个数据类型的标准化效果
    data_tuple = (
        comp_data,
        np.ones((3, 2)),
        np.ones((3, 1)),
        np.ones((2, 2))
    )
    
    transformed_data, scalers = fit_transform(data_tuple)
    comp_transformed = transformed_data[0]
    
    # 验证标准化后的数据均值接近0，标准差接近1
    assert abs(comp_transformed.mean()) < 1e-10  # 均值接近0
    assert abs(comp_transformed.std() - 1.0) < 1e-10  # 标准差接近1


def test_fit_transform_invalid_inputs():
    """测试异常输入"""
    # 空数组
    with pytest.raises(Exception):
        fit_transform((np.array([]), np.array([]), np.array([]), np.array([])))
    
    # 非数值数据
    with pytest.raises(Exception):
        fit_transform(("invalid", "data", "types", "here"))


def test_fit_transform_nan_handling():
    """测试NaN值的处理"""
    comp_data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    proc_data = np.array([[10.0, 20.0], [30.0, np.nan], [50.0, 60.0]])
    prop_data = np.array([[100.0], [200.0], [np.nan]])
    elem_feature = np.array([[0.1, 0.2], [np.nan, 0.4]])
    
    data_tuple = (comp_data, proc_data, prop_data, elem_feature)
    
    # 函数应该能够处理NaN值，或者抛出适当异常
    try:
        transformed_data, scalers = fit_transform(data_tuple)
        # 如果成功，验证输出形状
        assert transformed_data[0].shape == comp_data.shape
    except ValueError:
        # 如果scaler不支持NaN，这是预期的
        pass


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])