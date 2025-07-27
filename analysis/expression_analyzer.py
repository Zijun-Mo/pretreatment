"""
表情分析器
用于分析面部表情的激活强度
"""

from typing import Dict
from core.landmark_processor import LandmarkProcessor


class ExpressionAnalyzer:
    """表情分析器"""
    
    def __init__(self):
        self.landmark_processor = LandmarkProcessor()
        self.expression_names = {
            'eyebrow_raise': 0.0,
            'eye_close': 0.0,
            'nose_scrunch': 0.0,
            'smile': 0.0,
            'lip_pucker': 0.0,
            'neutral': 0.0
        }
    
    def analyze_expressions(self, detection_result, reference_landmarks = None) -> Dict[str, float]:
        """分析关键表情的激活强度"""
        face_blendshapes = detection_result.face_blendshapes[0]
        landmarks = detection_result.face_landmarks[0]
        if reference_landmarks is not None:
            aligned_landmarks = self.landmark_processor.align_landmarks(landmarks, reference_landmarks)
        expressions = self.expression_names.copy()
        
        # 将blendshapes转换为字典便于查找
        blendshape_dict = {}
        for blendshape in face_blendshapes:
            blendshape_dict[blendshape.category_name] = float(blendshape.score)
        
        # eyebrow_raise: browInnerUp, browOuterUpLeft, browOuterUpRight
        brow_components = ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight']
        brow_values = [blendshape_dict.get(comp, 0.0) for comp in brow_components]
        expressions['eyebrow_raise'] = max(brow_values)

        # eye_close: eyeBlinkLeft, eyeBlinkRight
        eye_components = ['eyeBlinkLeft', 'eyeBlinkRight']
        eye_values = [blendshape_dict.get(comp, 0.0) for comp in eye_components]
        expressions['eye_close'] = max(eye_values)

        # nose_scrunch:
        if reference_landmarks is not None:
            left_rest = self.landmark_processor.calc_distance(
                self.landmark_processor.get_point(reference_landmarks, 133),
                self.landmark_processor.get_point(reference_landmarks, 126))
            left_move = self.landmark_processor.calc_distance(
                self.landmark_processor.get_point(aligned_landmarks, 133),
                self.landmark_processor.get_point(aligned_landmarks, 126))
            right_rest = self.landmark_processor.calc_distance(
                self.landmark_processor.get_point(reference_landmarks, 362),
                self.landmark_processor.get_point(reference_landmarks, 355))
            right_move = self.landmark_processor.calc_distance(
                self.landmark_processor.get_point(aligned_landmarks, 362),
                self.landmark_processor.get_point(aligned_landmarks, 355))

            left_l = left_rest - left_move
            right_l = right_rest - right_move
            expressions['nose_scrunch'] = (left_l + right_l) / (left_rest + right_rest)
        else:
            # 如果没有参考地标，使用blendshape的值
            expressions['nose_scrunch'] = (blendshape_dict.get('noseSneerLeft', 0.0) + blendshape_dict.get('noseSneerRight', 0.0)) / 2.0

        # smile: mouthSmileLeft + mouthSmileRight + cheekSquintLeft + cheekSquintRight
        smile_components = ['mouthSmileLeft', 'mouthSmileRight', 'cheekSquintLeft', 'cheekSquintRight']
        smile_values = [blendshape_dict.get(comp, 0.0) for comp in smile_components]
        expressions['smile'] = sum(smile_values) / len(smile_components)

        # lip_pucker: mouthPucker, mouthFunnel
        pucker_components = ['mouthPucker', 'mouthFunnel']
        pucker_values = [blendshape_dict.get(comp, 0.0) for comp in pucker_components]
        expressions['lip_pucker'] = max(pucker_values)

        # 自定义neutral值：1 - (五个表情值的平方平均)
        five_expressions = [expressions['eyebrow_raise'], expressions['eye_close'], expressions['nose_scrunch'], expressions['smile'], expressions['lip_pucker']]
        five_expressions_rms = (sum([v**2 for v in five_expressions]) / 5.0) ** 0.5
        expressions['neutral'] = 1.0 - five_expressions_rms

        return expressions
