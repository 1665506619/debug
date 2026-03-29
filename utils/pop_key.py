import json

data = json.load(open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/chatrex.json'))

for d in data:
    for ann in d['annotation']:
        ann.pop('mask')
        ann.pop('point')

with open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/chatrex.json', 'w') as f:
    json.dump(data, f, indent=2)