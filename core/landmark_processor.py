"""
面部特征点处理工具
包含特征点计算、对齐等基础功能
"""

import numpy as np
import logging
import cv2
from typing import List, Tuple, Optional


class LandmarkProcessor:
    """处理人脸特征点的工具类"""
    
    def __init__(self):
        # 定义关键参考点索引（基于MediaPipe 468点模型）
        self.NOSE_TIP = 1         # 鼻尖
        self.NOSE_CENTER = 6      # 鼻梁中心
        self.LEFT_MOUTH_CORNER = 61  # 左嘴角
        self.RIGHT_MOUTH_CORNER = 291 # 右嘴角
        self.CHIN_CENTER = 18     # 下巴中心
        
        # 用于计算眼部中心的关键点（MediaPipe 468点模型）
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 398, 388, 387, 386, 385, 384, 466]
    
    def __get_eye_center(self, landmarks, is_left_eye: bool = True) -> np.ndarray:
        """计算眼部中心点，安全地处理索引"""
        eye_landmarks = self.LEFT_EYE_LANDMARKS if is_left_eye else self.RIGHT_EYE_LANDMARKS
        # 确保索引在范围内
        valid_landmarks = [idx for idx in eye_landmarks if idx < len(landmarks)]
        if not valid_landmarks:
            # 如果没有有效索引，返回默认位置
            return np.array([0.3 if is_left_eye else 0.7, 0.4])
        
        eye_points = np.array([self.get_point(landmarks, idx) for idx in valid_landmarks])
        return np.mean(eye_points, axis=0)
    
    @staticmethod
    def get_point(landmarks, index: int) -> np.ndarray:
        """从landmark取点坐标"""
        pt = landmarks[index]
        return np.array([pt.x, pt.y])
    
    @staticmethod
    def get_3D_point(landmarks, index: int) -> np.ndarray:
        """从landmark取点坐标"""
        pt = landmarks[index]
        return np.array([pt.x, pt.y, pt.z])
    
    @staticmethod
    def calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点间欧氏距离"""
        return np.linalg.norm(p1 - p2)
    
    @staticmethod
    def align_landmarks(source_landmarks, target_landmarks, stable_indices=None):
        """
        三维点阵对齐：可选仅用稳定关键点子集（如鼻梁、眉心、眼眶等）做刚性配准，支持 (x, y, z) 三维刚性变换
        Args:
            source_landmarks: 源landmarks列表
            target_landmarks: 目标landmarks列表
            stable_indices: 稳定点索引列表（如[6,197,195,5,4,1,19,94,9,8,168,33,133,362,263]），默认None表示用全部点
        Returns:
            对齐后的landmarks列表
        """
        try:
            def extract_points_3d(landmarks, indices=None):
                points = []
                if indices is None:
                    for landmark in landmarks:
                        pt = landmark
                        x = getattr(pt, 'x', 0.0)
                        y = getattr(pt, 'y', 0.0)
                        z = getattr(pt, 'z', 0.0)
                        points.append([x, y, z])
                else:
                    for idx in indices:
                        if idx < len(landmarks):
                            pt = landmarks[idx]
                            x = getattr(pt, 'x', 0.0)
                            y = getattr(pt, 'y', 0.0)
                            z = getattr(pt, 'z', 0.0)
                            points.append([x, y, z])
                return np.array(points)

            if stable_indices is None:
                stable_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                                  397, 365, 379, 378, 400, 277, 152, 148, 176, 149, 150, 136, 
                                  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, ]

            source_points = extract_points_3d(source_landmarks, stable_indices)
            target_points = extract_points_3d(target_landmarks, stable_indices)

            # 检查输入有效性
            if len(source_points) != len(target_points):
                logging.warning("源点和目标点数量不匹配")
                return source_landmarks
            if len(source_points) < 3:
                logging.warning("点数过少，无法进行对齐")
                return source_landmarks

            # 计算质心
            source_centroid = np.mean(source_points, axis=0)
            target_centroid = np.mean(target_points, axis=0)
            # 去质心
            source_centered = source_points - source_centroid
            target_centered = target_points - target_centroid
            # 计算缩放因子
            source_scale = np.sqrt(np.sum(source_centered ** 2))
            target_scale = np.sqrt(np.sum(target_centered ** 2))
            if source_scale < 1e-8 or target_scale < 1e-8:
                logging.warning("特征点过于集中，使用原始点")
                return source_landmarks
            # 标准化
            source_normalized = source_centered / source_scale
            target_normalized = target_centered / target_scale
            # 计算旋转矩阵（使用SVD，三维）
            H = np.dot(source_normalized.T, target_normalized)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            # 应用变换到所有源点
            all_points = np.array([[getattr(pt, 'x', 0.0), getattr(pt, 'y', 0.0), getattr(pt, 'z', 0.0)] for pt in source_landmarks])
            all_points_centered = all_points - source_centroid
            all_points_normalized = all_points_centered / source_scale
            aligned_points = np.dot(all_points_normalized, R) * target_scale + target_centroid
            # 创建对齐后的landmarks对象
            class AlignedLandmark:
                def __init__(self, x, y, z=0.0):
                    self.x = float(x)
                    self.y = float(y)
                    self.z = float(z)
            aligned_landmarks = []
            for i, (x, y, z) in enumerate(aligned_points):
                aligned_landmarks.append(AlignedLandmark(x, y, z))
            return aligned_landmarks
        except Exception as e:
            logging.error(f"对齐过程中出现错误: {e}，使用原始landmarks")
            return source_landmarks
    
    def detect_face_orientation(self, landmarks) -> dict:
        """
        检测面部朝向角度
        返回: {
            'yaw': 偏航角（左右转头），
            'pitch': 俯仰角（上下点头），
            'roll': 翻滚角（左右倾斜）
        }
        """
        try:
            # 安全地获取关键点坐标，确保索引在范围内
            def safe_get_point(idx):
                if idx < len(landmarks):
                    return self.get_point(landmarks, idx)
                else:
                    # 返回默认位置
                    return np.array([0.5, 0.5])
            
            nose_tip = safe_get_point(self.NOSE_TIP)
            nose_center = safe_get_point(self.NOSE_CENTER)
            left_eye = self.__get_eye_center(landmarks, True)
            right_eye = self.__get_eye_center(landmarks, False)
            left_mouth = safe_get_point(self.LEFT_MOUTH_CORNER)
            right_mouth = safe_get_point(self.RIGHT_MOUTH_CORNER)
            chin = safe_get_point(self.CHIN_CENTER)
            
            # 计算翻滚角（roll）- 基于眼部连线的倾斜
            eye_vector = right_eye - left_eye
            roll_angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
            
            # 计算偏航角（yaw）- 基于面部对称性
            # 使用鼻尖到面部中线的偏移量
            face_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_offset = nose_tip[0] - face_center_x
            face_width = np.abs(right_eye[0] - left_eye[0])
            yaw_ratio = nose_offset / face_width if face_width > 0 else 0
            yaw_angle = np.arcsin(np.clip(yaw_ratio * 2, -1, 1)) * 180 / np.pi
            
            # 计算俯仰角（pitch）- 基于鼻尖到眼部连线的垂直距离
            eye_center = (left_eye + right_eye) / 2
            nose_to_eye_vector = nose_tip - eye_center
            eye_to_chin_vector = chin - eye_center
            
            # 计算俯仰角的近似值
            face_height = np.linalg.norm(eye_to_chin_vector)
            if face_height > 0:
                pitch_ratio = nose_to_eye_vector[1] / face_height
                pitch_angle = np.arcsin(np.clip(pitch_ratio, -1, 1)) * 180 / np.pi
            else:
                pitch_angle = 0
            
            return {
                'yaw': yaw_angle,      # 左右转头角度
                'pitch': pitch_angle,  # 上下点头角度  
                'roll': roll_angle     # 左右倾斜角度
            }
            
        except Exception as e:
            logging.error(f"面部朝向检测失败: {e}")
            return {'yaw': 0, 'pitch': 0, 'roll': 0}
    
    def correct_face_orientation(self, landmarks, target_angles: dict = None) -> list:
        """
        校正面部朝向，使面部朝向摄像头
        
        Args:
            landmarks: 原始特征点
            target_angles: 目标角度，默认为 {'yaw': 0, 'pitch': 0, 'roll': 0}
        
        Returns:
            校正后的特征点列表
        """
        if target_angles is None:
            target_angles = {'yaw': 0, 'pitch': 0, 'roll': 0}
        
        try:
            # 检测当前面部朝向
            current_angles = self.detect_face_orientation(landmarks)
            
            # 计算需要的校正角度
            correction_yaw = target_angles['yaw'] - current_angles['yaw']
            correction_pitch = target_angles['pitch'] - current_angles['pitch']
            correction_roll = target_angles['roll'] - current_angles['roll']
            
            # 转换特征点为numpy数组
            points = np.array([[pt.x, pt.y, getattr(pt, 'z', 0)] for pt in landmarks])
            
            # 计算面部中心点（用作旋转中心）
            face_center = np.mean(points, axis=0)
            
            # 将点移到原点
            centered_points = points - face_center
            
            # 创建旋转矩阵（按ZYX顺序应用旋转）
            # Roll rotation (around Z axis)
            roll_rad = np.radians(correction_roll)
            Rz = np.array([
                [np.cos(roll_rad), -np.sin(roll_rad), 0],
                [np.sin(roll_rad), np.cos(roll_rad), 0],
                [0, 0, 1]
            ])
            
            # Pitch rotation (around X axis)
            pitch_rad = np.radians(correction_pitch)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0, np.sin(pitch_rad), np.cos(pitch_rad)]
            ])
            
            # Yaw rotation (around Y axis)
            yaw_rad = np.radians(correction_yaw)
            Ry = np.array([
                [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                [0, 1, 0],
                [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
            ])
            
            # 组合旋转矩阵
            R = Rz @ Ry @ Rx
            
            # 应用旋转
            rotated_points = np.dot(centered_points, R.T)
            
            # 移回原位置
            corrected_points = rotated_points + face_center
            
            # 创建校正后的landmarks对象
            class CorrectedLandmark:
                def __init__(self, x, y, z=0.0):
                    self.x = float(x)
                    self.y = float(y)
                    self.z = float(z)
            
            corrected_landmarks = []
            for i, (x, y, z) in enumerate(corrected_points):
                corrected_landmarks.append(CorrectedLandmark(x, y, z))
            
            return corrected_landmarks
            
        except Exception as e:
            logging.error(f"面部朝向校正失败: {e}")
            return landmarks

