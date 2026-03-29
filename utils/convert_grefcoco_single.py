import json

data = json.load(open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/grefcoco.json'))

for d in data:
    new_ann = []
    for ann in d['annotation']:
        if ' and ' in ann['text']:
            continue
        new_ann.append(ann)
    
    if len(new_ann)>0:
        d['annotation'] = new_ann 
with open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/grefcoco_single.json', 'w') as f:
    json.dump(data, f, indent=2)