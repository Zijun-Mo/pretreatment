"""
配置文件
包含系统的默认配置参数
"""

import os
from pathlib import Path


class Config:
    """配置类"""
    
    # MediaPipe模型路径
    # MediaPipe模型路径
    MODEL_PATH = '/home/jun/picture/face_landmarker_v2_with_blendshapes.task'
    
    # 默认输入输出目录
    INPUT_DIR = '/home/jun/picture/测试集'
    OUTPUT_DIR = '/home/jun/picture/output'
    
    # 支持的视频格式
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    # 表情关键词
    EXPRESSION_KEYS = ['抬眉', '闭眼', '皱鼻', '咧嘴笑', '撅嘴']
    
    # 处理参数
    VISIBLE_THRESHOLD = 0.001
    PEAK_THRESHOLD = 0.9  # 峰值的90%作为阈值
    
    # 文件命名参数
    FILE_ID_OFFSET = 50  # 患者序号加上此偏移量作为文件ID
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    @classmethod
    def validate_paths(cls):
        """验证路径是否存在"""
        if not os.path.exists(cls.MODEL_PATH):
            raise FileNotFoundError(f"MediaPipe模型文件不存在: {cls.MODEL_PATH}")
        
        # 创建输出目录
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        return True
