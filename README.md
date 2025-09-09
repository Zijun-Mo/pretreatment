# 面部表情分析系统

这是一个基于MediaPipe的面部表情分析系统，采用模块化设计，可以批量处理视频文件，分析面部表情并生成可视化结果。系统现已集成**面部朝向检测和校正功能**，可以自动校正头部朝向偏差，提供更准确的面部运动对称性分析。

**注意：本项目由原始的 `hello.py` 文件（1239行）重构而来，已成功拆分为多个独立的模块文件，按功能进行了合理的组织。拆分后的代码结构更加清晰，便于维护和扩展。**

## 新功能特性 🆕

### 面部朝向校正系统
- **自动朝向检测**: 检测面部的yaw（偏航）、pitch（俯仰）、roll（翻滚）角度
- **智能朝向校正**: 通过3D旋转变换将面部调整为正面朝向
- **增强分析精度**: 减少头部朝向变化对运动对称性评估的影响
- **实时校正验证**: 提供校正效果的量化评估

### 核心优势
- ✅ 支持轻到中度的头部朝向偏差校正（≤30度）
- ✅ 提高不同拍摄角度下的分析一致性
- ✅ 保持原始数据不变，生成校正后的新数据
- ✅ 提供详细的朝向统计和校正效果监控

## 模块化拆分概述

### 主要文件拆分

| 原始文件 | 拆分后的文件 | 行数 | 主要功能 |
|---------|-------------|------|---------|
| hello.py | **主入口文件** | | |
| | main.py | 42 | 主程序入口 |
| | config.py | 38 | 配置管理 |
| | analysis/facial_analysis_engine.py | 97 | 面部分析引擎 |
| | video/video_processor.py | 377 | 视频处理器 |
| | video/batch_processor.py | 50 | 批量处理器 |

### 功能模块拆分

| 模块目录 | 文件名 | 主要类/功能 | 原文件对应行数 |
|---------|-------|------------|---------------|
| **core/** | | **核心工具模块** | |
| | landmark_processor.py | LandmarkProcessor | 19-97 |
| **analysis/** | | **分析模块** | |
| | expression_analyzer.py | ExpressionAnalyzer | 100-148 |
| | movement_calculator.py | MovementCalculator | 151-253 |
| | facial_analysis_engine.py | FacialAnalysisEngine | 460-695 |
| **visualization/** | | **可视化模块** | |
| | font_manager.py | FontManager | 256-293 |
| | visualizers.py | BaseVisualizer, LandmarkVisualizer, SneerPointsVisualizer | 296-457 |
| | expression_visualizer.py | ExpressionVisualizer | 460-695 |
| **video/** | | **视频处理模块** | |
| | video_processor.py | VideoProcessor | 698-1150 |
| | batch_processor.py | BatchProcessor | 1150-1200 |
| **data/** | | **数据处理模块** | |
| | extractor.py | DataExtractor | 720-751 |

## 项目结构
```
face_expression_analysis/
├── main.py                     # 主入口文件
├── config.py                   # 配置文件
├── requirements.txt            # 依赖包列表
├── README.md                   # 项目说明文档
├── analysis/                   # 分析模块
│   ├── __init__.py
│   ├── expression_analyzer.py  # 表情分析器
│   ├── movement_calculator.py  # 运动幅度计算器
│   └── facial_analysis_engine.py # 面部分析引擎
├── core/                       # 核心工具模块
│   ├── __init__.py
│   └── landmark_processor.py   # 特征点处理器
├── video/                      # 视频处理模块
│   ├── video_processor.py      # 视频处理器
│   └── batch_processor.py      # 批量处理器
├── visualization/              # 可视化模块
│   ├── __init__.py
│   ├── font_manager.py         # 字体管理器  
│   ├── visualizers.py          # 基础可视化器
│   └── expression_visualizer.py # 表情可视化器
└── data/                       # 数据处理模块
    ├── __init__.py
    └── extractor.py            # 数据提取器
```

## 模块间依赖关系

```
main.py
├── config.py
├── video/batch_processor.py
│   └── video/video_processor.py
│       └── analysis/facial_analysis_engine.py
│           ├── analysis/
│           │   ├── expression_analyzer.py
│           │   └── movement_calculator.py → core/landmark_processor.py
│           ├── visualization/
│           │   ├── font_manager.py
│           │   ├── visualizers.py → font_manager.py
│           │   └── expression_visualizer.py → visualizers.py
│           └── data/extractor.py
```

## 模块说明

### 核心模块 (core/)
- **landmark_processor.py**: 处理人脸特征点，包括坐标提取、距离计算、特征点对齐等核心功能

### 分析模块 (analysis/)
- **expression_analyzer.py**: 基于blendshape数据分析面部表情，识别抬眉、闭眼、皱鼻、咧嘴笑、撅嘴等表情
- **movement_calculator.py**: 计算面部运动的对称性比率，分析左右两侧的运动幅度差异
- **facial_analysis_engine.py**: 面部分析引擎，整合各个分析和可视化模块

### 视频处理模块 (video/)
- **video_processor.py**: 视频处理器，负责视频的逐帧分析和结果输出
- **batch_processor.py**: 批量处理器，支持处理文件夹中的多个视频文件

### 可视化模块 (visualization/)
- **font_manager.py**: 管理中文字体，支持在图像上显示中文标注
- **visualizers.py**: 基础可视化器，包括特征点绘制、耸鼻子特征点标注等
- **expression_visualizer.py**: 表情信息可视化，在图像上绘制表情分析结果和运动比率

### 数据处理模块 (data/)
- **extractor.py**: 从MediaPipe检测结果中提取特征点和blendshape数据

## 使用方法

### 1. 安装依赖
```bash
pip install mediapipe opencv-python numpy pillow matplotlib
```

### 2. 配置路径
编辑 `config.py` 文件，设置：
- MediaPipe模型路径 (`MODEL_PATH`)
- 输入视频目录 (`INPUT_DIR`)
- 输出结果目录 (`OUTPUT_DIR`)

### 3. 运行程序
```bash
python main.py
```

### 4. 模块导入示例
```python
# 使用单个模块
from analysis.expression_analyzer import ExpressionAnalyzer
from core.landmark_processor import LandmarkProcessor

# 使用主要功能
from analysis.facial_analysis_engine import FacialAnalysisEngine
```

## 快速开始 - 面部朝向校正

### 基本使用示例
```python
from analysis.movement_calculator import MovementCalculator

# 创建分析器
movement_calc = MovementCalculator()

# 启用朝向校正的运动分析
movement_ratios = movement_calc.calculate_facial_movement_ratios(
    rest_landmarks,     # 静息态特征点
    move_landmarks,     # 运动态特征点
    correct_orientation=True  # 启用朝向校正 🆕
)

print(f"运动对称性比率: {movement_ratios}")
```

### 朝向检测示例
```python
from core.landmark_processor import LandmarkProcessor

processor = LandmarkProcessor()

# 检测面部朝向
orientation = processor.detect_face_orientation(landmarks)
print(f"面部朝向: yaw={orientation['yaw']:.1f}°, "
      f"pitch={orientation['pitch']:.1f}°, "
      f"roll={orientation['roll']:.1f}°")

# 校正面部朝向
corrected_landmarks = processor.correct_face_orientation(landmarks)
```

### 视频处理示例
```python
from video.video_processor import VideoProcessor

# 创建支持朝向校正的视频处理器
processor = VideoProcessor(enable_orientation_correction=True)

# 处理视频文件
result = processor.process_video_file("input.mp4", "output_with_correction.mp4")
```

## 模块化设计优势

### 1. **模块化设计**
- 每个模块职责单一，功能明确
- 便于团队协作开发
- 易于单元测试

### 2. **代码复用**
- 核心功能可以被多个模块使用
- 便于功能扩展和新增

### 3. **维护性提升**
- 修改某个功能时只需关注对应模块
- 降低了代码耦合度
- 便于调试和排错

### 4. **可扩展性**
- 新增表情识别算法时，只需扩展analysis模块
- 新增可视化效果时，只需扩展visualization模块
- 支持插件化架构

### 5. **配置管理**
- 统一的配置文件管理所有参数
- 便于部署和环境切换

## 功能特点

1. **表情识别**: 识别抬眉、闭眼、皱鼻、咧嘴笑、撅嘴等5种主要表情
2. **运动对称性分析**: 计算左右两侧面部运动的对称性比率
3. **可视化标注**: 在图像上绘制特征点、表情信息和分析结果
4. **批量处理**: 支持批量处理多个视频文件
5. **多格式支持**: 支持mp4、avi、mov等多种视频格式
6. **结果导出**: 生成标注图片、标注视频和JSON数据文件

## 输出结果

对于每个处理的视频，系统会生成：
- 基准图片（中性表情状态）
- 各表情的峰值图片（按表情分类）
- 标注视频（仅在表情峰值帧进行标注）
- JSON数据文件（包含详细的分析数据）

## 注意事项

1. 确保MediaPipe模型文件路径正确
2. 输入视频应包含清晰的人脸
3. 程序会自动创建输出目录
4. 建议使用GPU加速以提高处理速度

## 系统要求

- Python 3.7+
- Windows/Linux/macOS
- 建议配置GPU以提高处理速度

## 模块验证结果

所有模块导入测试已通过：
- ✅ 配置模块导入成功
- ✅ 核心模块导入成功  
- ✅ 分析模块导入成功
- ✅ 运动计算器模块导入成功
- ✅ 可视化模块导入成功
- ✅ 面部分析引擎导入成功
- ✅ 主入口文件导入成功

## 项目总结

通过模块化拆分，原本的单一大文件（hello.py，1239行）已经转换为结构清晰的多模块系统。每个模块都有明确的职责和界限，代码的可读性、可维护性和可扩展性都得到了显著提升。拆分后的系统保持了原有的所有功能，同时为后续的功能扩展和优化奠定了良好的基础。

### 主要改进：
- **代码结构更清晰**：按功能模块组织，便于理解和维护
- **便于团队协作**：不同开发者可以专注于不同的模块
- **提高可测试性**：每个模块可以独立进行单元测试
- **支持功能扩展**：新增功能时只需修改相关模块
- **降低耦合度**：模块间依赖关系清晰，便于重构和优化
