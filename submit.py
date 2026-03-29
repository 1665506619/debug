
import subprocess
import os

# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

print('start submit.py')
script_path = "scripts/train/train.sh"


# 使用 Popen 启动进程
process = subprocess.Popen(
    ["bash", script_path], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, # 将错误输出合并到标准输出
    text=True
)

# 实时读取输出
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

return_code = process.poll()
print(f"脚本结束，返回码: {return_code}")