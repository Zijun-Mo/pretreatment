"""
面部分析引擎模块
包含主要的业务逻辑和处理流程
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2



class FacialAnalysisEngine:
    """面部分析引擎 - 主要的业务逻辑类（集成ROI提取功能）"""

    def __init__(self, default_expand_ratio: float = 0.1):
        """初始化面部分析引擎"""
        self.logger = logging.getLogger(__name__)
        self.default_expand_ratio = default_expand_ratio

    def extract_face_roi(self, rgb_frame: np.ndarray, detection_result,
                        expand_ratio: float = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, int]]]:
        """
        从帧中提取面部ROI区域
        Args:
            rgb_frame: RGB格式的输入图像
            detection_result: MediaPipe检测结果
            expand_ratio: ROI扩展比例，用于在面部周围添加边距
        Returns:
            Tuple[face_roi, roi_info]: 面部ROI图像和ROI信息字典
        """
        if not detection_result.face_landmarks:
            self.logger.warning("未检测到面部特征点")
            return None, None

        face_landmarks = detection_result.face_landmarks[0]
        if expand_ratio is None:
            expand_ratio = self.default_expand_ratio

        height, width = rgb_frame.shape[:2]
        bbox = self._calculate_bbox(face_landmarks, width, height, expand_ratio)
        if bbox is None:
            return None, None
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        roi = rgb_frame[y:y+h, x:x+w]
        # 需要标注的特征点索引
        key_indices = [159, 145, 386, 374, 209, 429, 19, 0, 17, 61, 291]
        # 计算中点
        def midpoint(idx1, idx2):
            lm1 = face_landmarks[idx1]
            lm2 = face_landmarks[idx2]
            return ((lm1.x + lm2.x) / 2, (lm1.y + lm2.y) / 2)
        mid_105_52 = midpoint(105, 52)
        mid_334_282 = midpoint(334, 282)
        # 收集所有点的归一化坐标
        points = []
        # 中点
        points.append(mid_105_52)
        points.append(mid_334_282)
        # 其余点
        for idx in key_indices:
            lm = face_landmarks[idx]
            points.append((lm.x, lm.y))
        # 将所有点映射到ROI坐标系
        roi_points = []
        for px, py in points:
            roi_x = (px * rgb_frame.shape[1] - x) / w * 112
            roi_y = (py * rgb_frame.shape[0] - y) / h * 112
            roi_points.append((int(round(roi_x)), int(round(roi_y))))
        # 新建单通道图像用于画点
        points_img = np.zeros((112, 112), dtype=np.uint8)
        for pt in roi_points:
            cv2.circle(points_img, pt, 2, 255, -1)
        return roi, points_img, bbox

    def _calculate_bbox(self, face_landmarks, img_width: int, img_height: int,
                        expand_ratio: float) -> Optional[Dict[str, int]]:
        """计算面部边界框"""
        try:
            x_coords = [landmark.x * img_width for landmark in face_landmarks]
            y_coords = [landmark.y * img_height for landmark in face_landmarks]
            if not x_coords or not y_coords:
                return None
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            width = max_x - min_x
            height = max_y - min_y
            # 以人脸区域中心为中心，扩展为正方形
            side = max(width, height)
            expand_side = side * (1 + expand_ratio)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            x = int(center_x - expand_side / 2)
            y = int(center_y - expand_side / 2)
            # 边界检查
            x = max(0, x)
            y = max(0, y)
            w = min(img_width - x, int(expand_side))
            h = min(img_height - y, int(expand_side))
            return {'x': x, 'y': y, 'width': w, 'height': h}
        except Exception as e:
            self.logger.error(f"计算边界框失败: {e}")
            return None

    def process_frame_for_roi(self, rgb_frame: np.ndarray, detection_result):
        """
        处理帧并提取ROI（保持原有接口兼容性）
        
        Args:
            rgb_frame: RGB格式的输入图像
            detection_result: MediaPipe检测结果
            
        Returns:
            提取的面部ROI图像，如果未检测到面部则返回原图像
        """
        face_roi, points_roi, roi_info = self.extract_face_roi(rgb_frame, detection_result)
        if face_roi is not None:
            self.logger.info(f"成功提取面部ROI: {roi_info}")
            return face_roi, points_roi
        else:
            self.logger.error("未能提取面部ROI，返回原图像")
            return None, None
