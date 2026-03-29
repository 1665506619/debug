import json

# for i in range(5):
#     print(i)
data = json.load(open(f'/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/mapillary.json'))
for d in data:
    tmp = d['height']
    d['height'] = d['width']
    d['width'] = tmp
json.dump(data, open(f'/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/seg_train/seg-train/mapillary.json', 'w'))
    