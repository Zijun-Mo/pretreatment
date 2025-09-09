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


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.analysis_engine = FacialAnalysisEngine()
        self.expression_analyzer = ExpressionAnalyzer()
        self._init_mediapipe()
        # 表情名称映射（英文）
        self.expression_keys = ['eyebrow_raise', 'eye_close', 'nose_scrunch', 'smile', 'lip_pucker']
        # line_keys顺序与expression_keys一一对应，每个表情对应两个行号
        self.line_keys = [
            (9, 17),  # eyebrow_raise
            (10, 18),  # eye_close
            (11, 19),  # nose_scrunch
            (12, 20),  # smile
            (13, 21)   # lip_pucker
        ]
    
    def _init_mediapipe(self):
        """初始化MediaPipe"""
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
    
    def process_single_video(self, video_path: str, output_dir: str) -> bool:
        """处理单个视频文件"""
        try:
            video_name = Path(video_path).stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"处理视频: {video_name}")
            
            # 第一阶段：分析整个视频，找到基准帧和表情峰值帧
            baseline_frame, expression_peaks = self._analyze_video_for_peaks(video_path)
            
            if not baseline_frame:
                logging.error("未找到合适的基准帧")
                return False
            
            if not expression_peaks:
                logging.error("未找到表情峰值帧")
                return False
            
            # 第二阶段：保存基准图和表情图片，并获取标注好的帧
            self._save_baseline_and_expression_images(
                video_path, baseline_frame, expression_peaks, output_dir, video_name)
            return True
            
        except Exception as e:
            logging.error(f"处理视频 {video_path} 时出错: {e}")
            return False
    
    def _analyze_video_for_peaks(self, video_path: str):
        """分析整个视频，找到基准帧和表情峰值帧"""
        # 阶段一：通过分析中性表情找到基准帧
        print("第一阶段-A：初步分析视频，寻找基准帧...")

        # 加载动作时间戳
        video_name = Path(video_path).stem
        json_path = Path(video_path).parent / f"{video_name}_action_timestamps.json"
        if not json_path.exists():
            logging.error(f"时间戳文件未找到: {json_path}")
            return None, None
        with open(json_path, 'r', encoding='utf-8') as f:
            action_timestamps = json.load(f)['actions']

        # 直接使用英文表情名，无需映射
        
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
        
        first_pass_data = []
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
                    # 检查当前帧是否在任何一个动作的时间范围内
                    in_action_range = False
                    for action in action_timestamps.values():
                        if action['start_time'] <= timestamp_s <= action['end_time']:
                            in_action_range = True
                            break
                    
                    if in_action_range:
                        expressions = self.expression_analyzer.analyze_expressions(detection_result)
                        current_frame_info = {
                            'frame_number': frame_count,
                            'timestamp_ms': timestamp_ms,
                            'timestamp_s': timestamp_s,
                            'frame': frame.copy(),
                            'rgb_frame': rgb_frame.copy(),
                            'detection_result': detection_result,
                            'expressions': expressions
                        }
                        first_pass_data.append(current_frame_info)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"初步分析进度: {progress:.1f}% ({frame_count}/{total_frames})")
        cap.release()

        if not first_pass_data:
            logging.error("在指定的时间戳范围内未能找到任何有效的帧。")
            return None, None

        # 在 neutral 时间范围内寻找基准帧
        neutral_start_time = action_timestamps['neutral']['start_time']
        neutral_end_time = action_timestamps['neutral']['end_time']
        
        neutral_frames_data = [
            f for f in first_pass_data 
            if neutral_start_time <= f['timestamp_s'] <= neutral_end_time
        ]

        if not neutral_frames_data:
            logging.error("在中性表情时间范围内未找到任何帧。")
            return None, None

        # 计算每帧的中性状态
        neutral_values = [f['expressions']['neutral'] for f in neutral_frames_data]
        timestamps = [f['timestamp_s'] for f in neutral_frames_data]
        # 计算变化率（绝对值）
        neutral_deltas = [0.0]
        for i in range(1, len(neutral_values)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                delta = abs((neutral_values[i] - neutral_values[i-1]) / dt)
            else:
                delta = 0.0
            neutral_deltas.append(delta)
        # 前缀和
        prefix_sum = [0.0]
        for d in neutral_deltas:
            prefix_sum.append(prefix_sum[-1] + d)
        # 滑窗平均
        window_sec = 0.1
        window_size = max(1, int(window_sec * fps))
        min_avg = float('inf')
        min_idx = 0
        for i in range(len(neutral_deltas)):
            l = max(0, i - window_size)
            r = min(len(neutral_deltas)-1, i + window_size)
            count = r - l + 1
            avg = (prefix_sum[r+1] - prefix_sum[l]) / count
            if avg < min_avg:
                min_avg = avg
                min_idx = i
        
        baseline_frame_info = neutral_frames_data[min_idx].copy()
        print(f"基准帧已找到: 帧号 {baseline_frame_info['frame_number']} (中性值: {baseline_frame_info['expressions']['neutral']:.3f}, 变化率窗口均值: {min_avg:.5f})")
        
        # 阶段二：使用基准帧地标重新计算表情并找到峰值
        print("第一阶段-B：使用基准帧重新分析表情并寻找峰值...")
        reference_landmarks = baseline_frame_info['detection_result'].face_landmarks[0]
        
        final_frame_data = []
        expression_peaks = {expr: {'max_value': 0.0, 'frames': [], 'frameID': 0} for expr in self.expression_keys}

        for frame_info in first_pass_data:
            reanalyzed_expressions = self.expression_analyzer.analyze_expressions(
                frame_info['detection_result'], 
                reference_landmarks=reference_landmarks
            )
            
            # 使用新的表情更新帧信息并进行清理
            frame_info['expressions'] = reanalyzed_expressions
            frame_info['landmarks'] = frame_info['detection_result'].face_landmarks[0]
            frame_info['blendshapes'] = frame_info['detection_result'].face_blendshapes[0]
            
            final_frame_data.append(frame_info)
            
        # 使用新值更新表情峰值
        for expr_key in self.expression_keys:
            if expr_key in action_timestamps:
                time_range = action_timestamps[expr_key]
                for frame_info in final_frame_data:
                    if time_range['start_time'] <= frame_info['timestamp_s'] <= time_range['end_time']:
                        expr_value = frame_info['expressions'][expr_key]
                        if expr_value > expression_peaks[expr_key]['max_value']:
                            expression_peaks[expr_key]['max_value'] = expr_value
                            expression_peaks[expr_key]['frameID'] = frame_info['frame_number']
        
        # 找到每个表情峰值90%范围内的所有帧
        print("寻找表情峰值90%的帧...")
        for expr_key in self.expression_keys:
            if expr_key in action_timestamps:
                peak_value = expression_peaks[expr_key]['max_value']
                threshold_90 = peak_value * Config.PEAK_THRESHOLD

                print(f"{expr_key}: 帧号 = {expression_peaks[expr_key]['frameID']}, 峰值={peak_value:.3f}, 90%阈值={threshold_90:.3f}")

                time_range = action_timestamps[expr_key]
                qualifying_frames = []
                for frame_info in final_frame_data:
                    if (time_range['start_time'] <= frame_info['timestamp_s'] <= time_range['end_time'] and
                            frame_info['expressions'][expr_key] >= threshold_90):
                        qualifying_frames.append(frame_info)
                
                expression_peaks[expr_key]['frames'] = qualifying_frames
                print(f"找到{len(qualifying_frames)}帧达到{expr_key}的90%峰值")
        
        return baseline_frame_info, expression_peaks
    
    def _save_baseline_and_expression_images(self, video_path, baseline_frame, expression_peaks, output_dir, video_name):
        """保存基准图和表情图片"""
        print("第二阶段：保存基准图和表情图片...")
        # 保存基准图
        baseline_path = output_dir / f"{video_name}_baseline.jpg"
        points_baseline_path = output_dir / f"{video_name}_baseline_points.jpg"
        print(f"保存基准图: {baseline_path}")
        roi_baseline_frame, points_baseline_frame = self.analysis_engine.process_frame_for_roi(baseline_frame['frame'], baseline_frame['detection_result'])
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

        # 查找视频目录下的xlsx文件
        video_dir = Path(video_path).parent
        # 跳过以~$开头的临时/锁定xlsx文件
        xlsx_files = [f for f in video_dir.glob('*.xlsx') if not f.name.startswith('~$')]
        xlsx_df = None
        if xlsx_files:
            try:
                xlsx_df = pd.read_excel(xlsx_files[0], header=None)
            except Exception as e:
                print(f"读取xlsx失败: {e}")
                xlsx_df = None
        for expr_key in self.expression_keys:
            expr_frames = expression_peaks[expr_key]['frames']
            if not expr_frames:
                print(f"警告: {expr_key} 没有找到合适的帧")
                continue
            print(f"保存 {expr_key} 的 {len(expr_frames)} 张图片...")
            for i, frame_info in enumerate(expr_frames):
                # 保存roi区域和点图
                roi_frame, points_frame = self.analysis_engine.process_frame_for_roi(frame_info['frame'], frame_info['detection_result'])
                img_dir = output_dir / f"{expr_key}" / f"{video_name}_{i+1:03d}"
                img_path = img_dir / "facial_image.jpg"
                points_img_path = img_dir / "facial_image_points.jpg"
                # 确保目录存在
                img_dir.mkdir(parents=True, exist_ok=True)
                # 使用PIL保存图片
                try:
                    # 光流对比（Farneback）
                    # 先缩放到相同大小
                    if roi_frame.shape[:2] != roi_baseline_frame.shape[:2]:
                        roi_frame_resized = cv2.resize(roi_frame, (roi_baseline_frame.shape[1], roi_baseline_frame.shape[0]))
                    else:
                        roi_frame_resized = roi_frame
                    gray_base = cv2.cvtColor(roi_baseline_frame, cv2.COLOR_BGR2GRAY) if len(roi_baseline_frame.shape) == 3 else roi_baseline_frame
                    gray_cur = cv2.cvtColor(roi_frame_resized, cv2.COLOR_BGR2GRAY) if len(roi_frame_resized.shape) == 3 else roi_frame_resized
                    # 第一步：用光流将当前帧配准到基准帧
                    flow1 = cv2.calcOpticalFlowFarneback(gray_base, gray_cur, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                    h, w = gray_base.shape
                    flow_map = np.zeros_like(flow1, dtype=np.float32)
                    for y in range(h):
                        for x in range(w):
                            flow_map[y, x, 0] = x + flow1[y, x, 0]
                            flow_map[y, x, 1] = y + flow1[y, x, 1]
                    # 用flow_map将当前帧配准到基准帧
                    if len(roi_frame_resized.shape) == 3:
                        cur_for_warp = roi_frame_resized
                    else:
                        cur_for_warp = cv2.cvtColor(roi_frame_resized, cv2.COLOR_GRAY2BGR)
                    warped_cur = cv2.remap(cur_for_warp, flow_map[...,0], flow_map[...,1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    # 第二步：用配准后的帧与基准帧再计算光流
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
                    # 保存表情图和表情点图（resize到112x112）
                    rgb_image = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                    rgb_image_112 = cv2.resize(rgb_image, (112, 112), interpolation=cv2.INTER_LINEAR)
                    pil_image = Image.fromarray(rgb_image_112)
                    pil_image.save(str(img_path), 'JPEG', quality=95)
                    points_img_112 = cv2.resize(points_frame, (112, 112), interpolation=cv2.INTER_NEAREST)
                    points_img = Image.fromarray(points_img_112)
                    points_img.save(str(points_img_path), 'JPEG', quality=95)
                    # 保存基准图和基准点图到该目录（resize到112x112）
                    baseline_img_path = img_dir / "facial_image_baseline.jpg"
                    baseline_points_img_path = img_dir / "facial_image_baseline_points.jpg"
                    baseline_rgb = cv2.cvtColor(roi_baseline_frame, cv2.COLOR_BGR2RGB) if len(roi_baseline_frame.shape) == 3 else roi_baseline_frame
                    baseline_rgb_112 = cv2.resize(baseline_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
                    baseline_pil = Image.fromarray(baseline_rgb_112)
                    baseline_pil.save(str(baseline_img_path), 'JPEG', quality=95)
                    baseline_points_112 = cv2.resize(points_baseline_frame, (112, 112), interpolation=cv2.INTER_NEAREST)
                    baseline_points_pil = Image.fromarray(baseline_points_112)
                    baseline_points_pil.save(str(baseline_points_img_path), 'JPEG', quality=95)
                    # 保存xlsx文件line_keys行G列（每个表情保存对应的两行G列为json）
                    if xlsx_df is not None:
                        expr_idx = self.expression_keys.index(expr_key)
                        row1, row2 = self.line_keys[expr_idx]
                        g_dict = {}
                        try:
                            g_val = xlsx_df.iloc[row1, 6]  # G列
                        except Exception as e:
                            g_val = None
                        g_dict['dynamics'] = g_val
                        try:
                            g_val = xlsx_df.iloc[row2, 6]  # G列
                        except Exception as e:
                            g_val = None
                        g_dict['synkinesis'] = g_val
                        g_json_path = img_dir / "expression_g_column.json"
                        try:
                            with open(g_json_path, 'w', encoding='utf-8') as f:
                                json.dump(g_dict, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            print(f"保存G列json失败: {e}")
                except Exception as e:
                    print(f"保存图片失败: {e}")
                    logging.error(f"保存图片失败: {img_path}, 错误: {e}")
                    continue
            print(f"{expr_key} 图片保存完成，共 {len(expr_frames)} 张")
        return