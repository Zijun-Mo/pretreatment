"""
数据提取器
用于从MediaPipe结果中提取结构化数据
"""

from typing import List, Dict


class DataExtractor:
    """数据提取器"""
    
    @staticmethod
    def extract_landmarks_data(detection_result) -> List[List[Dict[str, float]]]:
        """提取人脸特征点数据"""
        landmarks_data = []
        
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                face_data = []
                for landmark in face_landmarks:
                    face_data.append({
                        'x': float(landmark.x),
                        'y': float(landmark.y)
                    })
                landmarks_data.append(face_data)
        
        return landmarks_data
    
    @staticmethod
    def extract_blendshapes_data(detection_result) -> List[Dict[str, float]]:
        """提取混合形状数据"""
        blendshapes_data = []
        
        if detection_result.face_blendshapes:
            for face_blendshapes in detection_result.face_blendshapes:
                shapes_data = {}
                for blendshape in face_blendshapes:
                    shapes_data[blendshape.category_name] = float(blendshape.score)
                blendshapes_data.append(shapes_data)
        
        return blendshapes_data
