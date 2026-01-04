import pandas as pd
import re

# 1. 加载数据
# 假设文件名仍为 'dax.xlsx - Sheet1.csv'，且没有表头
df = pd.read_excel('dax.xlsx', header=None)
raw_strings = df[0].astype(str)

def parse_ti_alloy(text):
    # 初始化结果字典
    composition = {}
    
    # 正则表达式说明：
    # (\d+\.?\d*) 匹配数字（如 10, 3.5, 0.15）
    # ([A-Z][a-z]?) 匹配元素符号（如 Mo, Cu, Fe, Sn, O, Al）
    # 注意：这里适配了数字在元素前的情况
    pattern = r'(\d+\.?\d*)([A-Z][a-z]?)'
    matches = re.findall(pattern, text)
    
    total_other = 0.0
    for value, element in matches:
        val = float(value)
        composition[element] = val
        total_other += val
    
    # 计算基底 Ti 的含量 (100 - 其他元素总和)
    composition['Ti'] = round(100.0 - total_other, 2)
    
    return composition

# 2. 执行解析
parsed_data = [parse_ti_alloy(s) for s in raw_strings]

# 3. 转换为 DataFrame 并整理列顺序
result_df = pd.DataFrame(parsed_data)

# 将 Ti 列移动到第一列
cols = ['Ti'] + [c for c in result_df.columns if c != 'Ti']
result_df = result_df[cols]

# 4. 填充空值为 0 并保存
result_df = result_df.fillna(0)
result_df.to_excel('titanium_alloys_extracted.xlsx', index=False)

print("提取并计算完成！前10行预览：")
print(result_df.head(10))