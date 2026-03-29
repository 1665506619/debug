import json
import re
import random
from pycocotools import mask as maskUtils
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
try:
    from refer import REFER
except ImportError:
    REFER = None

def singleMask2rle(mask):
    if mask is None:
        return None
    rle = maskUtils.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
    
def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


class SEG_DATA():
    def __init___(self):
        pass

    def MEVIS_VAL(self, ):
        print('begin')
        final_data = []
        meta_exp = json.load(open('/mnt/damovl/yuanyq/datasets/Sa2VA-Training/video_datas/mevis/train/valid_u/meta_expressions.json'))
        mask_dict = json.load(open('/mnt/damovl/yuanyq/datasets/Sa2VA-Training/video_datas/mevis/train/valid_u/mask_dict.json'))
        for vd in tqdm(meta_exp['videos']):
            video = meta_exp['videos'][vd]
            frame_num = len(video['frames'])
            dic = {}
            dic['video'] = 'mevis/train/valid_u/JPEGImages/'+vd
            dic['conversations'] = []
            masks = []
            for exp in video['expressions']:
                exps = video['expressions'][exp]
                expression = exps['exp']
                masks = []
                for idx in range(frame_num): 
                    mask = None
                    for i,obj in enumerate(exps['obj_id']):
                        annid = exps['anno_id'][i]
                        if mask_dict[str(annid)][idx] is not None:
                            if mask is None:
                                mask = annToMask(mask_dict[str(annid)][idx])
                            else:
                                mask = np.bitwise_or(mask, annToMask(mask_dict[str(annid)][idx]))
                    masks.append(singleMask2rle(mask))
                dic['expression'] = expression
                dic['masks'] = masks
                final_data.append(dic.copy())
                  
        print(len(final_data))
        b = json.dumps(final_data, indent=4)
        f2 = open(f'/mnt/workspace/workgroup/yuanyq/code/video_seg/vl3/evaluation/val_data/mevis_eval_seg.json', 'w')
        f2.write(b)
        f2.close()  

    
    def DAVIS_EVAL(self):
        final_data = []
        ann_path = '/mnt/damovl/yuanyq/datasets/Sa2VA-Training/video_datas/davis17/valid/Annotations/'
        meta_exp = json.load(open('/mnt/damovl/yuanyq/datasets/Sa2VA-Training/video_datas/davis17/meta_expressions/valid/meta_expressions.json'))
        for vd in tqdm(meta_exp['videos']):
            video = meta_exp['videos'][vd]
            frame_num = len(video['frames'])
            dic = {}
            dic['video'] = 'davis17/valid/JPEGImages/'+vd
            dic['conversations'] = []
            for exp in video['expressions']:
                exps = video['expressions'][exp]
                expression = exps['exp']
                obj = exps['obj_id']
                masks = []
                for idx in range(frame_num): 
                    mask = Image.open(os.path.join(ann_path, vd, video['frames'][idx]+'.png'))
                    mask = np.array(mask)==int(obj)
                    masks.append(singleMask2rle(mask))
                dic['expression'] = expression
                dic['masks'] = masks
                final_data.append(dic.copy())
                  
        print(len(final_data))
        b = json.dumps(final_data, indent=4)
        f2 = open(f'/mnt/workspace/workgroup/yuanyq/code/video_seg/vl3/evaluation/val_data/davis_eval.json', 'w')
        f2.write(b)
        f2.close()

    def ReasonSeg_EVAL(self, split='val'):
        print('begin')
        final_data = []
        data_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/ReasonSeg'
        output_dir = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval'
        
        split_dir = os.path.join(data_root, split)
        json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
        
        for json_file in tqdm(json_files):
            json_path = os.path.join(split_dir, json_file)
            annotation = json.load(open(json_path))
            
            # Get question from text field
            text = annotation.get('text', [])
            if not text or len(text) == 0:
                continue
            question = text[0] if isinstance(text, list) else text
            
            # Get image name
            shapes = annotation.get('shapes', [])
            if not shapes:
                continue
            
            image_name = shapes[0].get('image_name', json_file.replace('.json', '.jpg'))
            
            # Get image size and full path
            full_image_path = os.path.join(split_dir, image_name)
            if not os.path.exists(full_image_path):
                continue
            img = Image.open(full_image_path)
            h, w = img.size[1], img.size[0]
            
            # Process all target polygons and merge them
            target_polygons = []
            for shape in shapes:
                if shape.get('label') == 'target' and shape.get('shape_type') == 'polygon':
                    points = shape.get('points', [])
                    if points:
                        # Flatten points for RLE encoding
                        flat_points = [coord for point in points for coord in point]
                        target_polygons.append(flat_points)
            
            if not target_polygons:
                continue
            
            # Convert target polygons to RLE
            rles = maskUtils.frPyObjects(target_polygons, h, w)
            if len(rles) > 1:
                rle = maskUtils.merge(rles)
            else:
                rle = rles[0]
            
            # Process ignore polygons
            ignore_polygons = []
            for shape in shapes:
                if shape.get('label') == 'ignore' and shape.get('shape_type') == 'polygon':
                    points = shape.get('points', [])
                    if points:
                        flat_points = [coord for point in points for coord in point]
                        ignore_polygons.append(flat_points)
            
            dic = {
                'image': full_image_path,
                'question': question,
                'mask': {
                    'size': rle['size'],
                    'counts': rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
                }
            }
            
            # Add ignore mask if exists
            if ignore_polygons:
                ignore_rles = maskUtils.frPyObjects(ignore_polygons, h, w)
                if len(ignore_rles) > 1:
                    ignore_rle = maskUtils.merge(ignore_rles)
                else:
                    ignore_rle = ignore_rles[0]
                dic['ignore_mask'] = {
                    'size': ignore_rle['size'],
                    'counts': ignore_rle['counts'].decode('utf-8') if isinstance(ignore_rle['counts'], bytes) else ignore_rle['counts']
                }
            
            final_data.append(dic)
        
        print(len(final_data))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'reason_seg_{split}.json')
        b = json.dumps(final_data, indent=4)
        f2 = open(output_file, 'w')
        f2.write(b)
        f2.close()  

   

    def RynnEC_EVAL(self, split='object_segmentation'):
        print('begin')
        final_data = []
        data_root = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/RynnEC-Bench'
        output_dir = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval'
        
        json_file = os.path.join(data_root, f'{split}.json')
        if not os.path.exists(json_file):
            print(f'File not found: {json_file}')
            return
        
        annotations = json.load(open(json_file))
        
        for item in tqdm(annotations):
            dic = {}
            
            video_id = item.get('video_id')
            if not video_id:
                continue
            
            dic['video_root'] = f'data/{video_id}'
            dic['video'] = item.get('video', [])
            dic['conversations'] = item.get('conversations', [])
            dic['masks'] = item.get('masks', [])
            dic['mask_ids'] = item.get('mask_ids', [])
            dic['timestamps'] = item.get('timestamps', [])
            
            # Generate caption based on type for eval_rynnec.py get_type function
            task_type = item.get('type', 'direct referring')
            if 'situational' in task_type.lower():
                # For situational referring, use complex expression
                question = dic['conversations'][0]['value'] if dic['conversations'] else ''
                dic['caption'] = f'[complex expression] {question}'
            else:
                # For direct referring, use simple expression
                question = dic['conversations'][0]['value'] if dic['conversations'] else ''
                dic['caption'] = f'[simple expression] {question}'
            
            final_data.append(dic)
        
        print(len(final_data))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'rynnec_{split}.json')
        b = json.dumps(final_data, indent=4)
        f2 = open(output_file, 'w')
        f2.write(b)
        f2.close()

   

    def RefYouTubeVOS_EVAL(self):
        print('begin')
        final_data = []
        data_root = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/Refer-YouTube-VOS'
        output_dir = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval'
        
        ann_path = os.path.join(data_root, 'valid/Annotations')
        meta_file = os.path.join(data_root, 'valid/meta_expressions_challenge.json')
        meta_exp = json.load(open(meta_file))
        
        for vd in tqdm(meta_exp['videos']):
            video = meta_exp['videos'][vd]
            frames = video['frames']
            
            for exp_id in video['expressions']:
                exp_info = video['expressions'][exp_id]
                expression = exp_info['exp']
                obj_id = exp_info['obj_id']
                
                dic = {}
                dic['video'] = f'Refer-YouTube-VOS/valid/JPEGImages/{vd}'
                dic['expression'] = expression
                dic['frame_names'] = frames  # Add frame_names for correct evaluation
                
                masks = []
                h, w = None, None
                
                for frame_id in frames:
                    mask_file = os.path.join(ann_path, vd, obj_id, f'{frame_id}.png')
                    if os.path.exists(mask_file):
                        mask_img = Image.open(mask_file)
                        mask = np.array(mask_img) > 0
                        if h is None:
                            h, w = mask.shape
                        masks.append(singleMask2rle(mask.astype(np.uint8)))
                    else:
                        # If mask doesn't exist for this frame, get size and create zero mask
                        if h is None:
                            # Get size from first available mask or JPEGImage
                            for fid in frames:
                                test_mask = os.path.join(ann_path, vd, obj_id, f'{fid}.png')
                                if os.path.exists(test_mask):
                                    h, w = np.array(Image.open(test_mask)).shape
                                    break
                            if h is None:
                                # Get from JPEGImage
                                jpg_path = os.path.join(data_root, 'valid/JPEGImages', vd, f'{frames[0]}.jpg')
                                img = Image.open(jpg_path)
                                h, w = img.size[1], img.size[0]
                        masks.append(singleMask2rle(np.zeros((h, w), dtype=np.uint8)))
                
                dic['masks'] = masks
                final_data.append(dic.copy())
        
        print(len(final_data))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'ref_youtube_vos_valid.json')
        b = json.dumps(final_data, indent=4)
        f2 = open(output_file, 'w')
        f2.write(b)
        f2.close()

   

    def ReasonVOS_EVAL(self):
        print('begin')
        final_data = []
        data_root = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/revos'
        output_dir = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval'
        
        meta_exp = json.load(open(os.path.join(data_root, 'meta_expressions_valid_.json')))
        mask_dict = json.load(open(os.path.join(data_root, 'mask_dict.json')))
        
        for vd in tqdm(meta_exp['videos']):
            video = meta_exp['videos'][vd]
            frame_num = len(video['frames'])
            height = video['height']
            width = video['width']
            
            dic = {}
            dic['video'] = f'revos/{vd}'
            dic['video_id'] = vd
            dic['height'] = height
            dic['width'] = width
            
            for exp in video['expressions']:
                exps = video['expressions'][exp]
                expression = exps['exp']
                type_id = exps.get('type_id', 0)  # 0=referring, 1=reasoning
                masks = []
                
                for idx in range(frame_num):
                    mask = None
                    for i, obj in enumerate(exps['obj_id']):
                        annid = exps['anno_id'][i]
                        if mask_dict[str(annid)][idx] is not None:
                            if mask is None:
                                mask = annToMask(mask_dict[str(annid)][idx])
                            else:
                                mask = np.bitwise_or(mask, annToMask(mask_dict[str(annid)][idx]))
                    masks.append(singleMask2rle(mask))
                
                dic['expression'] = expression
                dic['type_id'] = type_id
                dic['exp_id'] = exp
                dic['masks'] = masks
                final_data.append(dic.copy())
        
        print(len(final_data))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'reason_vos_valid.json')
        b = json.dumps(final_data, indent=4)
        f2 = open(output_file, 'w')
        f2.write(b)
        f2.close()

   

    def ReasonVOS_EVAL_MINI(self, num_samples=10):
        print('begin - creating mini test set')
        final_data = []
        data_root = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/revos'
        output_dir = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval'
        
        meta_exp = json.load(open(os.path.join(data_root, 'meta_expressions_valid_.json')))
        mask_dict = json.load(open(os.path.join(data_root, 'mask_dict.json')))
        
        count = 0
        for vd in tqdm(meta_exp['videos']):
            if count >= num_samples:
                break
            
            video = meta_exp['videos'][vd]
            frame_num = len(video['frames'])
            dic = {}
            dic['video'] = f'revos/{vd}'
            dic['video_id'] = vd
            
            for exp in video['expressions']:
                exps = video['expressions'][exp]
                expression = exps['exp']
                type_id = exps.get('type_id', 0)
                masks = []
                
                for idx in range(frame_num):
                    mask = None
                    for i, obj in enumerate(exps['obj_id']):
                        annid = exps['anno_id'][i]
                        if mask_dict[str(annid)][idx] is not None:
                            if mask is None:
                                mask = annToMask(mask_dict[str(annid)][idx])
                            else:
                                mask = np.bitwise_or(mask, annToMask(mask_dict[str(annid)][idx]))
                    masks.append(singleMask2rle(mask))
                
                dic['expression'] = expression
                dic['type_id'] = type_id
                dic['exp_id'] = exp
                dic['masks'] = masks
                final_data.append(dic.copy())
                count += 1
                if count >= num_samples:
                    break
        
        print(len(final_data))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'reason_vos_mini_test.json')
        b = json.dumps(final_data, indent=4)
        f2 = open(output_file, 'w')
        f2.write(b)
        f2.close()

   

    def EgoMask_EVAL(self, split='long', num_samples=None):
        """
        处理 EgoMask 数据集
        split: 'long', 'medium', 'short', 'full'
        num_samples: 如果指定，只处理前 num_samples 个样本（用于测试）
        """
        print(f'开始处理 EgoMask {split} 数据集')
        if num_samples:
            print(f'测试模式：只处理前 {num_samples} 个样本')
        final_data = []
        
        # 数据路径
        annotation_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask'
        image_root = '/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data/EgoMask/EgoMask/dataset/egomask'
        output_dir = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval'
        
        # 根据 split 选择相应的 meta_expressions 文件
        if split == 'full':
            meta_file = os.path.join(annotation_root, 'meta_expressions.json')
        else:
            meta_file = os.path.join(annotation_root, f'subset/{split}/meta_expressions.json')
        
        if not os.path.exists(meta_file):
            print(f'Meta 文件不存在: {meta_file}')
            return
        
        meta_exp = json.load(open(meta_file))
        
        # 所有 split 的 annotations 都在同一个目录下
        annotations_dir = os.path.join(annotation_root, 'annotations')
        
        sample_count = 0
        for vid_name in tqdm(meta_exp['videos']):
            if num_samples and sample_count >= num_samples:
                break
                
            video = meta_exp['videos'][vid_name]
            frames = video['frames']
            
            # 判断视频类型来确定图片路径
            if split == 'short':
                # short subset 使用 refego
                video_path = f'EgoMask/EgoMask/dataset/egomask/JPEGImages/refego/{vid_name}'
            elif split == 'medium':
                # medium subset 的视频名称包含 "--"
                raw_clip_name = vid_name.split("--")[0]
                video_path = f'EgoMask/EgoMask/dataset/egomask/JPEGImages/egotracks/{raw_clip_name}'
            elif split == 'full':
                # full 需要根据 subset 字段判断
                vid_type = video.get('subset', 'long')
                if vid_type == 'short':
                    video_path = f'EgoMask/EgoMask/dataset/egomask/JPEGImages/refego/{vid_name}'
                else:
                    raw_clip_name = vid_name.split("--")[0] if '--' in vid_name else vid_name
                    video_path = f'EgoMask/EgoMask/dataset/egomask/JPEGImages/egotracks/{raw_clip_name}'
            else:
                # long subset
                video_path = f'EgoMask/EgoMask/dataset/egomask/JPEGImages/egotracks/{vid_name}'
            
            # 处理每个 expression
            for exp_id in video['expressions']:
                if num_samples and sample_count >= num_samples:
                    break
                    
                exp_info = video['expressions'][exp_id]
                expression = exp_info['exp']
                obj_id = exp_info['obj_id']
                
                dic = {}
                dic['video'] = video_path
                dic['expression'] = expression
                dic['video_id'] = vid_name
                dic['exp_id'] = str(exp_id)
                dic['obj_id'] = obj_id
                
                # 加载 ground truth masks (RLE 格式)
                # 对于 medium split，annotations 目录使用基础的 video_id（不带 --start--length 后缀）
                # 对于 short split，video_id 本身就不带后缀
                annot_vid_name = vid_name.split("--")[0] if '--' in vid_name else vid_name
                mask_file = os.path.join(annotations_dir, annot_vid_name, f'{obj_id}.json')
                
                # 关键修复：生成所有连续帧的 masks 数组（像 MeVIS/DAVIS 一样）
                # 获取视频目录下的所有帧
                video_full_path = os.path.join('/lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/data', video_path)
                
                # 检查视频目录是否存在
                if not os.path.exists(video_full_path):
                    print(f'警告: 视频目录不存在，跳过: {video_full_path}')
                    continue
                
                try:
                    all_frames = sorted([f.replace('.jpg', '') for f in os.listdir(video_full_path) if f.endswith('.jpg')])
                except Exception as e:
                    print(f'警告: 读取视频目录失败，跳过: {video_full_path}, 错误: {e}')
                    continue
                
                if len(all_frames) == 0:
                    print(f'警告: 视频目录为空，跳过: {video_full_path}')
                    continue
                
                # 【关键修复】使用 meta_expressions 中的 frames 字段（官方标注的帧）
                # 这样可以确保：
                # 1. 帧编号与GT mask的key完全匹配
                # 2. Medium/Long使用正确的绝对帧号
                # 3. Short使用正确的img前缀帧号
                meta_frames = video.get('frames', [])
                if meta_frames:
                    # 使用 meta_expressions 中的 frames（已经是正确格式）
                    all_frames = sorted(meta_frames)
                    print(f'{split} video {vid_name}: 使用meta_expressions中的 {len(all_frames)} 帧')
                else:
                    # 如果meta中没有frames字段，降级到从目录读取
                    print(f'警告: meta_expressions中没有frames字段，从目录读取: {vid_name}')
                
                # 保存所有帧的名称（用于后续加载对应的图像）
                dic['frame_names'] = all_frames
                
                masks = []
                if os.path.exists(mask_file):
                    gold_mask_rle = json.load(open(mask_file))
                    # 为所有连续帧创建 mask (如果该帧没有 GT mask，则为 None)
                    for frame_name in all_frames:
                        if frame_name in gold_mask_rle:
                            masks.append(gold_mask_rle[frame_name])
                        else:
                            masks.append(None)
                else:
                    print(f'警告: 未找到 mask 文件: {mask_file}')
                    masks = [None] * len(all_frames)
                
                dic['masks'] = masks
                
                final_data.append(dic.copy())
                sample_count += 1
        
        print(f'处理完成，共 {len(final_data)} 条数据')
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果是测试模式，文件名添加 _test 后缀
        if num_samples:
            output_file = os.path.join(output_dir, f'egomask_{split}_test.json')
        else:
            output_file = os.path.join(output_dir, f'egomask_{split}.json')
            
        b = json.dumps(final_data, indent=4)
        f2 = open(output_file, 'w')
        f2.write(b)
        f2.close()
        print(f'数据已保存到: {output_file}')

    def EgoMask_TRAIN(self, max_frames=16, num_obj=5, output_dir='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval'):
        """
        构建EgoMask训练集数据
        Args:
            max_frames: 每个视频均匀采样的最大帧数
            num_obj: 每个样本包含的物体数量
            output_dir: 输出目录
        """
        print('开始构建EgoMask训练集...')
        
        data_root = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask-train'
        
        # 读取meta expressions
        meta_exp_path = os.path.join(data_root, 'meta_expressions/train/meta_expressions.json')
        print(f'读取meta_expressions: {meta_exp_path}')
        meta_exp = json.load(open(meta_exp_path))
        
        # 读取meta (objects信息)
        meta_path = os.path.join(data_root, 'train/meta.json')
        print(f'读取meta: {meta_path}')
        meta = json.load(open(meta_path))
        
        # 读取mask_dict
        mask_dict_path = os.path.join(data_root, 'train/mask_dict.json')
        print(f'读取mask_dict: {mask_dict_path} (可能需要一些时间...)')
        mask_dict = json.load(open(mask_dict_path))
        
        final_data = []
        videos = meta_exp['videos']
        
        for vid_name in tqdm(videos, desc='处理视频'):
            video = videos[vid_name]
            expressions = video.get('expressions', {})
            all_frames = sorted(video.get('frames', []))
            
            if not all_frames or not expressions:
                continue
            
            # 均匀采样max_frames帧
            num_frames = len(all_frames)
            if num_frames <= max_frames:
                sampled_indices = list(range(num_frames))
            else:
                step = num_frames / max_frames
                sampled_indices = [int(i * step) for i in range(max_frames)]
            
            frame_idx = sampled_indices
            sampled_frames = [all_frames[i] for i in sampled_indices]
            
            # 获取所有expression IDs
            exp_ids = list(expressions.keys())
            
            # 随机采样num_obj个expressions（如果不够就重复采样）
            if len(exp_ids) >= num_obj:
                selected_exp_ids = random.sample(exp_ids, num_obj)
            else:
                # 不够就重复采样
                selected_exp_ids = random.choices(exp_ids, k=num_obj)
            
            # 构建conversations和masks
            conversations = []
            masks_list = []
            
            for exp_id in selected_exp_ids:
                exp_data = expressions[str(exp_id)]
                exp_text = exp_data.get('exp', '')
                obj_id = exp_data.get('obj_id', '')
                
                # 添加对话
                conversations.append({
                    "from": "human",
                    "value": f"Can you segment {exp_text} in the video?"
                })
                conversations.append({
                    "from": "gpt",
                    "value": "It is [SEG]."
                })
                
                # 获取该物体的masks
                # mask_dict的key是字符串格式：str(tuple)
                mask_key = str((vid_name, str(obj_id)))
                if mask_key in mask_dict:
                    obj_masks = mask_dict[mask_key]
                    
                    # 构建frame_idx -> mask_rle的字典
                    mask_dict_for_obj = {}
                    for i, frame_idx_val in enumerate(frame_idx):
                        frame_name = all_frames[frame_idx_val]
                        # 在obj_masks中找到对应帧的mask
                        # obj_masks是一个list，需要根据frame_name找到对应的mask
                        # 假设obj_masks的顺序与meta['videos'][vid_name]['objects'][obj_id]['frames']一致
                        obj_frames = meta['videos'][vid_name]['objects'].get(str(obj_id), {}).get('frames', [])
                        if frame_name in obj_frames:
                            obj_frame_idx = obj_frames.index(frame_name)
                            if obj_frame_idx < len(obj_masks) and obj_masks[obj_frame_idx] is not None:
                                mask_dict_for_obj[str(i)] = obj_masks[obj_frame_idx]
                            else:
                                mask_dict_for_obj[str(i)] = None
                        else:
                            mask_dict_for_obj[str(i)] = None
                    
                    masks_list.append(mask_dict_for_obj)
                else:
                    # 没有找到mask，全部设为None
                    mask_dict_for_obj = {str(i): None for i in range(len(frame_idx))}
                    masks_list.append(mask_dict_for_obj)
            
            # 确定video_path
            # 需要找到视频帧所在的目录
            video_path = self._get_video_path_train(vid_name, data_root)
            
            if video_path is None:
                print(f'警告: 未找到视频 {vid_name} 的帧目录，跳过')
                continue
            
            dic = {
                'video': video_path,
                'conversations': conversations,
                'frame_idx': frame_idx,
                'masks': masks_list
            }
            
            final_data.append(dic)
        
        print(f'处理完成，共 {len(final_data)} 条数据')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f'EGOMASK_{max_frames}f.json')
        b = json.dumps(final_data, indent=4)
        f2 = open(output_file, 'w')
        f2.write(b)
        f2.close()
        print(f'数据已保存到: {output_file}')
    
    def _get_video_path_train(self, vid_name, data_root):
        """
        找到训练集视频帧所在的相对目录
        Args:
            vid_name: 视频ID
            data_root: EgoMask训练集根目录
        Returns:
            相对路径（相对于/lustre/.../data/），如果找不到返回None
        """
        # 训练集视频帧的实际位置
        base_data_dir = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data'
        
        possible_paths = [
            f'EgoMask/EgoMask/dataset/egomask-train/train/JPEGImages/{vid_name}',
            f'refego/JPEGImages/{vid_name}',
            f'egotracks/{vid_name}',
        ]
        
        for rel_path in possible_paths:
            full_path = os.path.join(base_data_dir, rel_path)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                return rel_path
        
        return None


   

data = SEG_DATA()

# data.MEVIS_VAL()
# data.DAVIS_EVAL()
# data.ReasonSeg_EVAL(split='val')
# data.ReasonSeg_EVAL(split='test')
# data.RynnEC_EVAL(split='object_segmentation')
data.RefYouTubeVOS_EVAL()
# data.ReasonVOS_EVAL()
# data.ReasonVOS_EVAL_MINI(num_samples=10)

# EgoMask 完整数据集（取消注释需要的）
# data.EgoMask_EVAL(split='long')
# data.EgoMask_EVAL(split='medium')
# data.EgoMask_EVAL(split='short')  # 完整 short subset
# data.EgoMask_EVAL(split='full')

# EgoMask 测试模式（小数据集，用于快速测试）
# data.EgoMask_EVAL(split='short', num_samples=5)  # 只处理 5 个样本

# EgoMask 训练集构建
# data.EgoMask_TRAIN(max_frames=16, num_obj=5)