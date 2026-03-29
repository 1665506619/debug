import json
import glob 
from collections import defaultdict
final_data = []
data_list = glob.glob('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/eval/ours/*.json')
nums = defaultdict(int)
for d_ in data_list:
    data = json.load(open(d_))
    for d in data:
        for ann in d['annotation']:
            if 'mask' not in ann:
                nums[0] += 1
                continue
            nums[len(ann['mask'])] += 1
    final_data.extend(data)
print(nums)

with open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/eval/ours/ego_merge.json', 'w') as f:
    json.dump(final_data, f, indent=4)