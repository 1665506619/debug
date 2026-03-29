import json

final_data = []
for i in range(8):
    data_path = f'/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/humanref_45k_mask/shard_0{i}.jsonl'
    for d_ in open(data_path):
        data = json.loads(d_)
        data.pop('_global_idx')
        data.pop('_ok')
        new_ann = []
        for ann in data['annotation']:
            if ann['mask'] is None:
                print('none')
                continue
            ann.pop('bbox')
            ann.pop('point')

            new_ann.append(ann)
        data['annotation'] = new_ann

        final_data.append(data)
print(len(final_data))
with open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/humanref_45k_mask.json', 'w') as f:
    json.dump(final_data, f, indent=2)