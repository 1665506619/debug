import os
import json
import pandas as pd

# 定义文件夹路径
folder_path = "/mnt/workspace/workgroup/yuanyq/code/video_seg/vl3/evaluation_results/bak-10200-lora"  # 替换为你的文件夹路径

# 初始化结构存放数据
data = {
    "refCOCO": {"val": None, "testA": None, "testB": None},
    "refCOCO+": {"val": None, "testA": None, "testB": None},
    "refCOCOg": {"val": None, "test": None},
}

# 遍历文件夹下的JSON文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):  # 只处理JSON文件
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            ciou = round(content[0]["ciou"]*100,1)
        
        # 提取关键信息并归类
        if "refcoco_val" in file_name:
            data["refCOCO"]["val"] = ciou
        elif "refcoco_testA" in file_name:
            data["refCOCO"]["testA"] = ciou
        elif "refcoco_testB" in file_name:
            data["refCOCO"]["testB"] = ciou
        elif "refcoco+_val" in file_name:
            data["refCOCO+"]["val"] = ciou
        elif "refcoco+_testA" in file_name:
            data["refCOCO+"]["testA"] = ciou
        elif "refcoco+_testB" in file_name:
            data["refCOCO+"]["testB"] = ciou
        elif "refcocog_val" in file_name:
            data["refCOCOg"]["val"] = ciou
        elif "refcocog_test" in file_name:
            data["refCOCOg"]["test"] = ciou

print(data)
# # 构造DataFrame
# df = pd.DataFrame({
#     "refCOCO(ciou)": [data["refCOCO"]["val"], data["refCOCO"]["testA"], data["refCOCO"]["testB"]],
#     "refCOCO+(ciou)": [data["refCOCO+"]["val"], data["refCOCO+"]["testA"], data["refCOCO+"]["testB"]],
#     "refCOCOg(ciou)": [data["refCOCOg"]["val"], data["refCOCOg"]["test"]],
# }, index=["val", "testA", "testB"])

# # 删除多余的行 (只有两行的列保持一致)
# df.dropna(how='all', axis=0, inplace=True)

# # 保存到Excel
# output_path = folder_path+"ciou_summary.xlsx"  # 输出路径
# df.to_excel(output_path, index=True, index_label="Dataset Type")

# print(f"汇总结果已保存到 {output_path}")
