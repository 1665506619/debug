import json
data=json.load(open('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/seg-sam3/evaluation_results/0108_pretrain_v7_wo_none_multi_objs_lora/checkpoint-11933/grefcoco_val.json'))
nums = {}

for d in data:
    mask_rle = d.get('mask_rle')

    if mask_rle is None:
        nums[0] = nums.get(0, 0) + 1
        continue

    nums[len(mask_rle)] = nums.get(len(mask_rle), 0) + 1
print(nums)