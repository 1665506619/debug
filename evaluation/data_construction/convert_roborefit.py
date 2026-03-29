import json

data_list = [
    '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/roborefit_testA_qa.json',
    '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval/roborefit_testB_qa.json'
]

for data_path in data_list:
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {data_path}")
    for d in data:
        quetion = d['question']
        if quetion.lower().startswith(('will ', 'would ', 'can ', 'could ')) and not quetion.strip().endswith('?'):
            d['question'] = quetion.strip() + '?'
        else:
            d['question'] = quetion.strip() + '.'
        d['question'] = d['question'][0].upper() + d['question'][1:]

    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Processed and saved {len(data)} samples to {data_path}")