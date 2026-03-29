import json
from collections import defaultdict

data_list = [
    "ego4d_annotations.json",
    "epic_annotations.json",
    "paco_annotations.json",
    "sam_annotations.json"
]
nums = defaultdict(int)

final_data = []
final_num = 0
for d_ in data_list:
    data = json.load(open(f'/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/eval/ours/{d_}'))
    for d in data:
        if 'sa_' in d['image']:
            d['image'] = 'SA-1B/images/' + d['image']
        d['data_source'] = d_.split('_')[0]
        for ann in d['annotation']:
            if 'mask' not in ann:
                nums[0] += 1
                final_num += 1
                continue
            nums[len(ann['mask'])] += 1
            final_num += 1
    final_data.extend(data)

print(nums)
print(len(final_data))
print(final_num)
with open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/eval/ours/merge.json', 'w') as f:
    json.dump(final_data, f, indent=4)