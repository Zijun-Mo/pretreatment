"""
视频处理器模块
负责视频的分析、处理和输出
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
from config import Config
import pandas as pd

from analysis.facial_analysis_engine import FacialAnalysisEngine
from analysis.expression_analyzer import ExpressionAnalyzer


def safe_json_serialize(obj):
    """安全的JSON序列化函数，处理numpy类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.analysis_engine = FacialAnalysisEngine()
        self.expression_analyzer = ExpressionAnalyzer()
        self._init_mediapipe()
        # 表情名称映射（中文到英文）
        self.expression_mapping = {
            'action_raise_eyebrow': 'eyebrow_raise',
            'action_close_eyes_soft': 'eye_close', 
            'action_shrug_nose': 'nose_scrunch',
            'action_bare_teeth': 'smile',
            'action_pout': 'lip_pucker'
        }
        # 表情名称（英文）
        self.expression_keys = ['eyebrow_raise', 'eye_close', 'nose_scrunch', 'smile', 'lip_pucker']
        # 表情名称（中文）用于显示
        self.expression_names_zh = ['抬眉', '闭眼', '皱鼻', '咧嘴笑', '撅嘴']
        # 表情对应的Excel列索引 (动态列, 联动列)
        self.expression_col_mapping = {
            'eyebrow_raise': {'dynamic': 5, 'synkinesis': 11},   # 抬眉：动态E列(5)，联动K列(11)
            'eye_close': {'dynamic': 6, 'synkinesis': 12},       # 轻闭眼：动态F列(6)，联动L列(12)
            'nose_scrunch': {'dynamic': 7, 'synkinesis': 13},    # 皱鼻：动态G列(7)，联动M列(13)
            'smile': {'dynamic': 8, 'synkinesis': 14},           # 咧嘴笑：动态H列(8)，联动N列(14)
            'lip_pucker': {'dynamic': 9, 'synkinesis': 15}       # 撅嘴：动态I列(9)，联动O列(15)
        }
    
    def _init_mediapipe(self):
        """初始化MediaPipe"""
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
    
    def _get_expression_from_filename(self, filename: str) -> str:
        """从文件名获取表情类型"""
        for action_name, expression_key in self.expression_mapping.items():
            if action_name in filename:
                return expression_key
        return None
    
    def _get_patient_id_from_path(self, video_path: Path) -> str:
        """从路径获取患者序号"""
        # 从父目录名获取患者序号，如 "01", "02" 等
        parent_name = video_path.parent.name
        # 检查是否为数字格式
        if parent_name.isdigit():
            return parent_name
        return None
    
    def _analyze_single_expression_video(self, video_path: str, expression_type: str, baseline_frame: Optional[Dict] = None):
        """分析单个表情视频，使用统一的基准帧或第一帧作为基准帧，找到表情峰值帧"""
        print(f"分析视频 {Path(video_path).name}，表情类型: {expression_type}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"无法打开视频文件 {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        
        all_frames_data = []
        frame_count = 0
        
        with self.FaceLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_count * 1000 / fps)
                timestamp_s = frame_count / fps
                
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if detection_result.face_blendshapes and detection_result.face_landmarks:
                    # 如果没有提供基准帧，使用第一帧作为基准帧
                    if baseline_frame is None and frame_count == 0:
                        expressions = self.expression_analyzer.analyze_expressions(detection_result)
                        baseline_frame = {
                            'frame_number': frame_count,
                            'timestamp_ms': timestamp_ms,
                            'timestamp_s': timestamp_s,
                            'frame': frame.copy(),
                            'rgb_frame': rgb_frame.copy(),
                            'detection_result': detection_result,
                            'expressions': expressions,
                            'landmarks': detection_result.face_landmarks[0]
                        }
                        print(f"基准帧已设置: 帧号 {frame_count} (中性值: {expressions['neutral']:.3f})")
                    
                    # 记录所有帧数据
                    frame_info = {
                        'frame_number': frame_count,
                        'timestamp_ms': timestamp_ms,
                        'timestamp_s': timestamp_s,
                        'frame': frame.copy(),
                        'rgb_frame': rgb_frame.copy(),
                        'detection_result': detection_result,
                        'landmarks': detection_result.face_landmarks[0]
                    }
                    all_frames_data.append(frame_info)
                
                frame_count += 1
                if frame_count % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"分析进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        
        if not baseline_frame or not all_frames_data:
            logging.error("未能获取有效的帧数据")
            return None, None
        
        # 使用基准帧地标重新分析所有帧的表情
        print("使用基准帧重新分析表情...")
        reference_landmarks = baseline_frame['landmarks']
        
        max_expression_value = 0.0
        peak_frames = []
        
        for frame_info in all_frames_data:
            # 重新分析表情
            expressions = self.expression_analyzer.analyze_expressions(
                frame_info['detection_result'], 
                reference_landmarks=reference_landmarks
            )
            frame_info['expressions'] = expressions
            
            # 查找目标表情的峰值
            expr_value = expressions[expression_type]
            if expr_value > max_expression_value:
                max_expression_value = expr_value
        
        # 找到达到峰值90%的所有帧
        threshold_90 = max_expression_value * Config.PEAK_THRESHOLD
        print(f"{expression_type}: 峰值={max_expression_value:.3f}, 90%阈值={threshold_90:.3f}")
        
        for frame_info in all_frames_data:
            expr_value = frame_info['expressions'][expression_type]
            if expr_value >= threshold_90:
                peak_frames.append(frame_info)
        
        print(f"找到 {len(peak_frames)} 帧达到 {expression_type} 的90%峰值")
        
        return baseline_frame, peak_frames
    
    def _select_unified_baseline_frame(self, video_paths: List[str]) -> Optional[Dict]:
        """
        从多个表情视频的第一帧中选择统一的基准帧
        选择5种表情值平均最低的第一帧作为基准帧
        """
        print("开始选择统一基准帧...")
        
        candidate_frames = []
        current_timestamp = 0
        
        for video_path in video_paths:
            print(f"提取视频第一帧: {Path(video_path).name}")
            
            # 为每个视频创建独立的landmarker实例
            options = self.FaceLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path=self.model_path),
                running_mode=self.VisionRunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1)
            
            with self.FaceLandmarker.create_from_options(options) as landmarker:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logging.warning(f"无法打开视频文件 {video_path}")
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # 读取第一帧
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"无法读取视频第一帧: {video_path}")
                    cap.release()
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = 0  # 每个视频都从0开始
                
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if detection_result.face_blendshapes and detection_result.face_landmarks:
                    # 分析表情
                    expressions = self.expression_analyzer.analyze_expressions(detection_result)
                    
                    # 计算5种表情值的平均
                    expr_values = [expressions[key] for key in self.expression_keys]
                    avg_expression = sum(expr_values) / len(expr_values)
                    
                    candidate_frame = {
                        'video_path': video_path,
                        'frame_number': 0,
                        'timestamp_ms': 0,
                        'timestamp_s': 0.0,
                        'frame': frame.copy(),
                        'rgb_frame': rgb_frame.copy(),
                        'detection_result': detection_result,
                        'expressions': expressions,
                        'landmarks': detection_result.face_landmarks[0],
                        'avg_expression': avg_expression
                    }
                    
                    candidate_frames.append(candidate_frame)
                    print(f"  表情值: {expressions}")
                    print(f"  平均表情值: {avg_expression:.4f}")
                
                cap.release()
        
        if not candidate_frames:
            logging.error("没有找到有效的候选基准帧")
            return None
        
        # 选择平均表情值最低的帧作为基准帧
        baseline_frame = min(candidate_frames, key=lambda x: x['avg_expression'])
        
        print(f"选择统一基准帧: {Path(baseline_frame['video_path']).name}")
        print(f"基准帧平均表情值: {baseline_frame['avg_expression']:.4f}")
        print(f"基准帧表情值: {baseline_frame['expressions']}")
        
        return baseline_frame
    
    def process_patient_videos(self, video_paths: List[str], output_dir: str, score_df: pd.DataFrame = None) -> bool:
        """
        处理一个患者的所有表情视频（使用统一基准帧）
        
        Args:
            video_paths: 该患者的所有表情视频路径列表
            output_dir: 输出目录
            score_df: 评分数据框
        """
        try:
            if not video_paths:
                logging.error("视频路径列表为空")
                return False
            
            print(f"开始处理患者的 {len(video_paths)} 个表情视频...")
            
            # 选择统一的基准帧
            unified_baseline_frame = self._select_unified_baseline_frame(video_paths)
            if not unified_baseline_frame:
                logging.error("无法选择统一的基准帧")
                return False
            
            # 获取患者序号（从第一个视频路径获取）
            patient_id = self._get_patient_id_from_path(Path(video_paths[0]))
            if not patient_id:
                logging.error("无法确定患者序号")
                return False
            
            print(f"患者序号: {patient_id}")
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            
            # 使用统一基准帧处理每个表情视频
            for video_path in video_paths:
                video_path_obj = Path(video_path)
                video_name = video_path_obj.stem
                
                print(f"\n处理视频: {video_name}")
                
                # 从文件名确定表情类型
                expression_type = self._get_expression_from_filename(video_name)
                if not expression_type:
                    logging.warning(f"无法从文件名 {video_name} 确定表情类型，跳过")
                    continue
                
                print(f"检测到表情类型: {expression_type}")
                
                # 使用统一基准帧分析视频
                baseline_frame, expression_peak_frames = self._analyze_single_expression_video(
                    video_path, expression_type, unified_baseline_frame)
                
                if not baseline_frame:
                    logging.warning(f"视频 {video_name} 未找到合适的基准帧，跳过")
                    continue
                
                if not expression_peak_frames:
                    logging.warning(f"视频 {video_name} 未找到表情峰值帧，跳过")
                    continue
                
                # 保存基准图和表情图片
                self._save_images_new_structure(
                    video_path, baseline_frame, expression_peak_frames, 
                    output_dir, video_name, expression_type, patient_id, score_df)
                
                success_count += 1
            
            print(f"\n患者 {patient_id} 处理完成，成功处理 {success_count}/{len(video_paths)} 个视频")
            return success_count > 0
            
        except Exception as e:
            logging.error(f"处理患者视频时出错: {e}")
            return False
    
    def _save_images_new_structure(self, video_path, baseline_frame, expression_peak_frames, 
                                   output_dir, video_name, expression_type, patient_id, score_df):
        """保存基准图和表情图片（新结构）"""
        print(f"保存基准图和表情图片 - {expression_type}...")
        
        # 生成新的文件名：患者序号+配置的偏移量
        new_file_id = str(int(patient_id) + Config.FILE_ID_OFFSET)
        
        # 保存基准图
        baseline_path = output_dir / f"{new_file_id}_baseline.jpg"
        points_baseline_path = output_dir / f"{new_file_id}_baseline_points.jpg"
        
        roi_baseline_frame, points_baseline_frame = self.analysis_engine.process_frame_for_roi(
            baseline_frame['frame'], baseline_frame['detection_result'])
        
        if roi_baseline_frame is None:
            print("基准图ROI提取失败,无法保存")
            return
        
        # 保存基准图（resize到112x112）
        if len(roi_baseline_frame.shape) == 3:
            rgb_image = cv2.cvtColor(roi_baseline_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = roi_baseline_frame
        rgb_image_112 = cv2.resize(rgb_image, (112, 112), interpolation=cv2.INTER_LINEAR)
        pil_image = Image.fromarray(rgb_image_112)
        pil_image.save(str(baseline_path), 'JPEG', quality=95)
        print(f"基准图已保存: {baseline_path}")
        
        # 保存基准点图（resize到112x112）
        points_img_112 = cv2.resize(points_baseline_frame, (112, 112), interpolation=cv2.INTER_NEAREST)
        points_img = Image.fromarray(points_img_112)
        points_img.save(str(points_baseline_path), 'JPEG', quality=95)
        print(f"基准点图已保存: {points_baseline_path}")
        
        print(f"基准图中性值: {baseline_frame['expressions']['neutral']:.3e}")
        
        # 保存表情图片
        if not expression_peak_frames:
            print(f"警告: {expression_type} 没有找到合适的帧")
            return
        
        print(f"保存 {expression_type} 的 {len(expression_peak_frames)} 张图片...")
        
        for i, frame_info in enumerate(expression_peak_frames):
            # 保存roi区域和点图
            roi_frame, points_frame = self.analysis_engine.process_frame_for_roi(
                frame_info['frame'], frame_info['detection_result'])
            
            img_dir = output_dir / f"{expression_type}" / f"{new_file_id}_{i+1:03d}"
            
            # 确保目录存在
            img_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 光流计算
                if roi_frame.shape[:2] != roi_baseline_frame.shape[:2]:
                    roi_frame_resized = cv2.resize(roi_frame, (roi_baseline_frame.shape[1], roi_baseline_frame.shape[0]))
                else:
                    roi_frame_resized = roi_frame
                
                gray_base = cv2.cvtColor(roi_baseline_frame, cv2.COLOR_BGR2GRAY) if len(roi_baseline_frame.shape) == 3 else roi_baseline_frame
                gray_cur = cv2.cvtColor(roi_frame_resized, cv2.COLOR_BGR2GRAY) if len(roi_frame_resized.shape) == 3 else roi_frame_resized
                
                # 计算光流
                flow1 = cv2.calcOpticalFlowFarneback(gray_base, gray_cur, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                h, w = gray_base.shape
                flow_map = np.zeros_like(flow1, dtype=np.float32)
                for y in range(h):
                    for x in range(w):
                        flow_map[y, x, 0] = x + flow1[y, x, 0]
                        flow_map[y, x, 1] = y + flow1[y, x, 1]
                
                # 配准
                if len(roi_frame_resized.shape) == 3:
                    cur_for_warp = roi_frame_resized
                else:
                    cur_for_warp = cv2.cvtColor(roi_frame_resized, cv2.COLOR_GRAY2BGR)
                warped_cur = cv2.remap(cur_for_warp, flow_map[...,0], flow_map[...,1], 
                                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                
                # 再次计算光流
                gray_warped = cv2.cvtColor(warped_cur, cv2.COLOR_BGR2GRAY) if len(warped_cur.shape) == 3 else warped_cur
                flow2 = cv2.calcOpticalFlowFarneback(gray_base, gray_warped, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                mag, ang = cv2.cartToPolar(flow2[...,0], flow2[...,1])
                mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                
                hsv = np.zeros((gray_base.shape[0], gray_base.shape[1], 3), dtype=np.uint8)
                hsv[...,0] = ang * 180 / np.pi / 2
                hsv[...,1] = 255
                hsv[...,2] = mag_norm.astype(np.uint8)
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                flow_bgr_112 = cv2.resize(flow_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
                optical_flow_path = img_dir / "optical_flow.jpg"
                cv2.imwrite(str(optical_flow_path), flow_bgr_112)
                    
                # 提取和保存表情图特征点张量
                expression_landmarks = frame_info['landmarks']
                expression_landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in expression_landmarks])
                expression_landmarks_path = img_dir / "expression_landmarks.npy"
                np.save(str(expression_landmarks_path), expression_landmarks_array)
                
                # 保存基准图特征点张量
                baseline_landmarks = baseline_frame['landmarks']
                baseline_landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in baseline_landmarks])
                baseline_landmarks_path = img_dir / "baseline_landmarks.npy"
                np.save(str(baseline_landmarks_path), baseline_landmarks_array)
                
                # 保存评分信息
                if score_df is not None:
                    cols = self.expression_col_mapping[expression_type]
                    g_dict = {}
                    
                    # 查找患者对应的行（患者序号在第1列，从第2行开始是数据）
                    patient_row_idx = None
                    patient_num = int(patient_id)
                    
                    for idx in range(len(score_df)):  # 从第1行开始查找（包含表头后的数据行）
                        try:
                            if pd.notna(score_df.iloc[idx, 0]) and int(score_df.iloc[idx, 0]) == patient_num:
                                patient_row_idx = idx
                                break
                        except (ValueError, TypeError):
                            continue
                    
                    if patient_row_idx is not None:
                        try:
                            g_val = score_df.iloc[patient_row_idx, cols['dynamic']]  # 动态评分列
                            # 使用安全序列化函数转换类型
                            g_val = safe_json_serialize(g_val)
                        except Exception as e:
                            g_val = None
                        g_dict['dynamics'] = g_val
                        
                        try:
                            g_val = score_df.iloc[patient_row_idx, cols['synkinesis']]  # 联动评分列
                            # 使用安全序列化函数转换类型
                            g_val = safe_json_serialize(g_val)
                        except Exception as e:
                            g_val = None
                        g_dict['synkinesis'] = g_val
                    else:
                        print(f"未找到患者{patient_id}的评分数据")
                        g_dict['dynamics'] = None
                        g_dict['synkinesis'] = None
                    
                    g_json_path = img_dir / "expression_g_column.json"
                    try:
                        with open(g_json_path, 'w', encoding='utf-8') as f:
                            json.dump(g_dict, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"保存G列json失败: {e}")
                
            except Exception as e:
                print(f"保存图片失败: {e}")
                logging.error(f"保存图片失败: {img_dir}, 错误: {e}")
                continue
        
        print(f"{expression_type} 图片保存完成，共 {len(expression_peak_frames)} 张")
    
    def process_single_video(self, video_path: str, output_dir: str, score_df: pd.DataFrame = None) -> bool:
        """
        处理单个视频文件（兼容性方法）
        
        注意：此方法使用单个视频的第一帧作为基准帧。
        推荐使用 process_patient_videos() 方法来处理一个患者的所有表情视频，
        该方法会选择最佳的统一基准帧。
        """
        try:
            video_path_obj = Path(video_path)
            video_name = video_path_obj.stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"处理视频: {video_name}")
            
            # 从文件名确定表情类型
            expression_type = self._get_expression_from_filename(video_name)
            if not expression_type:
                logging.error(f"无法从文件名 {video_name} 确定表情类型")
                return False
            
            print(f"检测到表情类型: {expression_type}")
            
            # 获取患者序号
            patient_id = self._get_patient_id_from_path(video_path_obj)
            if not patient_id:
                logging.error(f"无法从路径 {video_path} 确定患者序号")
                return False
            
            print(f"患者序号: {patient_id}")
            
            # 分析视频：第一帧作为基准帧，找到表情峰值帧
            baseline_frame, expression_peak_frames = self._analyze_single_expression_video(video_path, expression_type)
            
            if not baseline_frame:
                logging.error("未找到合适的基准帧")
                return False
            
            if not expression_peak_frames:
                logging.error("未找到表情峰值帧")
                return False
            
            # 保存基准图和表情图片
            self._save_images_new_structure(
                video_path, baseline_frame, expression_peak_frames, 
                output_dir, video_name, expression_type, patient_id, score_df)
            
            return True
            
        except Exception as e:
            logging.error(f"处理视频 {video_path} 时出错: {e}")
            return False