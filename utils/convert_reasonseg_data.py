import json

for split in ['test', 'val']:
    data = json.load(open(f'/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/eval/reason_seg_{split}.json'))
    for d in data:
        d['question'] = d['question'].replace('What is', 'Where is')
    with open(f'/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/eval/reason_seg_{split}.json', 'w') as f:
        json.dump(data, f, indent=2)