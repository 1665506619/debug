import json
import random
from pycocotools import mask as maskUtils
from tqdm import tqdm
import numpy as np
import os


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


class EgoMaskTrainDiverse():
    """EgoMask训练集构建（支持多样性采样）"""
    
    def _sample_frames_with_diversity(self, num_frames, target_frames, sampling_type='start'):
        """
        采样帧，支持多样性采样
        Args:
            num_frames: 视频总帧数
            target_frames: 目标采样帧数 (8, 16, 32, 64)
            sampling_type: 'start' (从头), 'middle' (中间), 'end' (从末)
        Returns:
            sampled_indices: 采样帧的索引列表
        """
        if num_frames <= target_frames:
            # 如果视频帧数小于等于目标帧数，使用所有帧
            return list(range(num_frames))
        
        # 计算步长：每step帧采样一帧
        step = num_frames / target_frames
        
        if sampling_type == 'start':
            # 从头开始采样: 0, step, 2*step, ...
            # 例如64帧采样8帧：0, 8, 16, 24, 32, 40, 48, 56
            start_offset = 0
            sampled_indices = [int(start_offset + i * step) for i in range(target_frames)]
            # 确保不越界
            sampled_indices = [min(idx, num_frames - 1) for idx in sampled_indices]
        elif sampling_type == 'middle':
            # 从中间开始采样: step/2, step/2+step, ...
            # 例如64帧采样8帧：4, 12, 20, 28, 36, 44, 52, 60
            start_offset = int(step / 2)
            sampled_indices = [int(start_offset + i * step) for i in range(target_frames)]
            # 确保不越界
            sampled_indices = [min(idx, num_frames - 1) for idx in sampled_indices]
        elif sampling_type == 'end':
            # 从末尾开始采样: 从索引step开始，使得最后一帧尽可能接近视频末尾
            # 例如64帧采样8帧：从索引step开始，8, 16, 24, 32, 40, 48, 56
            # 计算最大起始偏移，使得最后一帧不超过num_frames-1
            max_start_offset = int(num_frames - 1 - (target_frames - 1) * step)
            # 使用step作为起始偏移（如果不超过max_start_offset）
            start_offset = min(int(step), max(0, max_start_offset))
            sampled_indices = [int(start_offset + i * step) for i in range(target_frames)]
            # 确保不越界
            sampled_indices = [min(idx, num_frames - 1) for idx in sampled_indices]
        else:
            raise ValueError(f"Unknown sampling_type: {sampling_type}")
        
        # 去重并排序，保持顺序
        sampled_indices = sorted(list(set(sampled_indices)))
        
        # 如果采样到的帧数不够，补充帧
        if len(sampled_indices) < target_frames:
            # 补充缺失的帧
            all_indices = set(sampled_indices)
            available_indices = [i for i in range(num_frames) if i not in all_indices]
            needed = target_frames - len(sampled_indices)
            if available_indices:
                additional = random.sample(available_indices, min(needed, len(available_indices)))
                sampled_indices.extend(additional)
                sampled_indices = sorted(sampled_indices)
        
        return sampled_indices[:target_frames]
    
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
    
    def build(self, num_obj=5, output_dir='/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/eval', max_videos=None, output_suffix=''):
        """
        构建EgoMask训练集数据（支持多样性采样）
        Args:
            num_obj: 每个样本包含的物体数量
            output_dir: 输出目录
            max_videos: 最大处理视频数量（None表示处理所有视频，用于测试）
            output_suffix: 输出文件名后缀（用于区分测试和正式数据）
        """
        print('开始构建EgoMask训练集（多样性采样）...')
        if max_videos:
            print(f'测试模式：只处理前 {max_videos} 个视频')
        
        data_root = '/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data/EgoMask/egomask-train'
        
        # 读取meta expressions
        meta_exp_path = os.path.join(data_root, 'meta_expressions/train/meta_expressions.json')
        print(f'读取meta_expressions: {meta_exp_path}')
        if not os.path.exists(meta_exp_path):
            print(f'错误: 文件不存在 {meta_exp_path}')
            return
        meta_exp = json.load(open(meta_exp_path))
        
        # 读取meta (objects信息)
        meta_path = os.path.join(data_root, 'train/meta.json')
        print(f'读取meta: {meta_path}')
        if not os.path.exists(meta_path):
            print(f'错误: 文件不存在 {meta_path}')
            return
        meta = json.load(open(meta_path))
        
        # 读取mask_dict
        mask_dict_path = os.path.join(data_root, 'train/mask_dict.json')
        print(f'读取mask_dict: {mask_dict_path} (可能需要一些时间...)')
        if not os.path.exists(mask_dict_path):
            print(f'错误: 文件不存在 {mask_dict_path}')
            return
        mask_dict = json.load(open(mask_dict_path))
        
        final_data = []
        videos = meta_exp['videos']
        
        # 如果指定了max_videos，只处理前max_videos个视频
        if max_videos:
            video_items = list(videos.items())[:max_videos]
            videos = dict(video_items)
            print(f'限制处理视频数量: {len(videos)} 个')
        
        # 采样配置：目标帧数和采样类型
        target_frames_list = [8, 16, 32, 64]
        sampling_types = ['start', 'middle', 'end']
        
        sample_id = 0
        
        for vid_name in tqdm(videos, desc='处理视频'):
            video = videos[vid_name]
            expressions = video.get('expressions', {})
            all_frames = sorted(video.get('frames', []))
            
            if not all_frames or not expressions:
                continue
            
            num_frames = len(all_frames)
            
            # 获取所有expression IDs
            exp_ids = list(expressions.keys())
            if len(exp_ids) == 0:
                continue
            
            # 对每个采样配置生成样本
            for target_frames in target_frames_list:
                # 如果视频帧数太少，跳过较大的target_frames
                if num_frames < target_frames:
                    continue
                
                for sampling_type in sampling_types:
                    # 采样帧
                    sampled_indices = self._sample_frames_with_diversity(num_frames, target_frames, sampling_type)
                    
                    if len(sampled_indices) == 0:
                        continue
                    
                    frame_idx = sampled_indices
                    sampled_frames = [all_frames[i] for i in sampled_indices]
                    
                    # 随机采样num_obj个expressions（如果不够就重复采样）
                    if len(exp_ids) >= num_obj:
                        selected_exp_ids = random.sample(exp_ids, num_obj)
                    else:
                        selected_exp_ids = random.choices(exp_ids, k=num_obj)
                    
                    # 构建conversations和masks
                    conversations = []
                    masks_list = []
                    
                    for exp_id in selected_exp_ids:
                        exp_data = expressions[str(exp_id)]
                        exp_text = exp_data.get('exp', '')
                        obj_id = exp_data.get('obj_id', '')
                        
                        # 添加human问题
                        conversations.append({
                            "from": "human",
                            "value": f"Can you segment {exp_text} in the video?"
                        })
                        
                        # 获取该物体的masks
                        mask_key = str((vid_name, str(obj_id)))
                        mask_dict_for_obj = {}
                        has_any_mask = False
                        
                        if mask_key in mask_dict:
                            obj_masks = mask_dict[mask_key]
                            obj_frames = meta['videos'][vid_name]['objects'].get(str(obj_id), {}).get('frames', [])
                            
                            # 构建frame_idx值 -> mask_rle的字典（键使用frame_idx中的值，如"0", "37", "75"）
                            for i, frame_idx_val in enumerate(frame_idx):
                                frame_name = all_frames[frame_idx_val]
                                # 使用frame_idx中的值作为键
                                frame_key = str(frame_idx_val)
                                
                                if frame_name in obj_frames:
                                    obj_frame_idx = obj_frames.index(frame_name)
                                    if obj_frame_idx < len(obj_masks) and obj_masks[obj_frame_idx] is not None:
                                        mask_dict_for_obj[frame_key] = obj_masks[obj_frame_idx]
                                        has_any_mask = True
                                    else:
                                        mask_dict_for_obj[frame_key] = None
                                else:
                                    mask_dict_for_obj[frame_key] = None
                        else:
                            # 没有找到mask，全部设为None（键使用frame_idx中的值）
                            for frame_idx_val in frame_idx:
                                frame_key = str(frame_idx_val)
                                mask_dict_for_obj[frame_key] = None
                        
                        # 根据是否有mask决定回答
                        if has_any_mask:
                            conversations.append({
                                "from": "gpt",
                                "value": "It is [SEG]."
                            })
                        else:
                            # 所有帧都没有mask，回答"There isn't a xxx in this video."
                            conversations.append({
                                "from": "gpt",
                                "value": f"There isn't a {exp_text} in this video."
                            })
                        
                        masks_list.append(mask_dict_for_obj)
                    
                    # 确定video_path
                    video_path = self._get_video_path_train(vid_name, data_root)
                    
                    if video_path is None:
                        continue
                    
                    # 生成文件名：vid_name_target_frames_sampling_type_sample_id
                    filename = f"{vid_name}_{target_frames}f_{sampling_type}_{sample_id:06d}.json"
                    
                    dic = {
                        'video': video_path,
                        'conversations': conversations,
                        'frame_idx': frame_idx,
                        'masks': masks_list,
                        'filename': filename,  # 添加文件名字段
                        'video_id': vid_name,
                        'target_frames': target_frames,
                        'sampling_type': sampling_type
                    }
                    
                    final_data.append(dic)
                    sample_id += 1
        
        print(f'处理完成，共 {len(final_data)} 条数据')
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为单个JSON文件
        output_file = os.path.join(output_dir, f'EGOMASK_train_diverse{output_suffix}.json')
        b = json.dumps(final_data, indent=4)
        f2 = open(output_file, 'w')
        f2.write(b)
        f2.close()
        print(f'数据已保存到: {output_file} (共 {len(final_data)} 条)')
        
        # 同时按配置分组保存
        for target_frames in target_frames_list:
            for sampling_type in sampling_types:
                filtered_data = [d for d in final_data if d.get('target_frames') == target_frames and d.get('sampling_type') == sampling_type]
                if len(filtered_data) > 0:
                    output_file_grouped = os.path.join(output_dir, f'EGOMASK_{target_frames}f_{sampling_type}{output_suffix}.json')
                    b = json.dumps(filtered_data, indent=4)
                    f2 = open(output_file_grouped, 'w')
                    f2.write(b)
                    f2.close()
                    print(f'分组数据已保存到: {output_file_grouped} ({len(filtered_data)} 条)')


if __name__ == '__main__':
    # 创建实例并运行
    builder = EgoMaskTrainDiverse()
    builder.build(num_obj=5)

