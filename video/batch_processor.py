"""
批量处理器模块
负责批量处理多个视频文件
"""

from pathlib import Path
import pandas as pd
import numpy as np
from video.video_processor import VideoProcessor


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, model_path: str):
        self.video_processor = VideoProcessor(model_path)
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """批量处理文件夹中的所有视频"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 查找总评分表格文件
        score_df = self._load_score_table(input_path)
        
        # 查找所有患者文件夹
        patient_folders = [f for f in input_path.iterdir() if f.is_dir() and f.name.isdigit()]
        patient_folders.sort(key=lambda x: int(x.name))
        
        if not patient_folders:
            print(f"在目录 {input_dir} 中未找到患者文件夹")
            return
        
        print(f"找到 {len(patient_folders)} 个患者文件夹")
        
        total_videos = 0
        success_count = 0
        
        # 处理每个患者文件夹
        for patient_folder in patient_folders:
            print(f"\n处理患者文件夹: {patient_folder.name}")
            
            # 查找该患者文件夹下的所有视频
            video_files = set()
            for ext in self.video_extensions:
                video_files.update(patient_folder.glob(f"*{ext}"))
                video_files.update(patient_folder.glob(f"*{ext.upper()}"))
            
            video_files = list(video_files)
            
            if not video_files:
                print(f"  患者 {patient_folder.name} 文件夹中未找到视频文件")
                continue
            
            print(f"  找到 {len(video_files)} 个视频文件")
            
            # 使用新的统一基准帧方法处理该患者的所有视频
            video_paths = [str(video_file) for video_file in video_files]
            total_videos += len(video_files)
            
            print(f"  开始处理患者 {patient_folder.name} 的 {len(video_files)} 个视频（使用统一基准帧）")
            
            try:
                if self.video_processor.process_patient_videos(
                    video_paths, str(output_path), score_df):
                    success_count += len(video_files)
                    print(f"    成功处理患者 {patient_folder.name} 的所有视频")
                else:
                    print(f"    处理失败: 患者 {patient_folder.name}")
            except Exception as e:
                print(f"    处理出错: 患者 {patient_folder.name} - {e}")
        
        print(f"\n批量处理完成！")
        print(f"成功处理: {success_count}/{total_videos} 个视频")
    
    def _load_score_table(self, input_path: Path) -> pd.DataFrame:
        """加载评分表格 - 读取三个医生的CSV文件并计算平均分"""
        # 查找三个医生的CSV文件
        doctor_files = []
        for i in range(1, 4):  # doctor_1, doctor_2, doctor_3
            csv_file = input_path / f"doctor_{i}_scores.csv"
            if csv_file.exists():
                doctor_files.append(csv_file)
            else:
                print(f"警告: 未找到 {csv_file.name}")
        
        if len(doctor_files) == 0:
            print("警告: 未找到任何医生评分文件")
            return None
        
        print(f"找到 {len(doctor_files)} 个医生评分文件")
        
        # 读取所有医生的评分数据
        doctor_dataframes = []
        for csv_file in doctor_files:
            try:
                df = pd.read_csv(csv_file, header=0, encoding='utf-8')
                doctor_dataframes.append(df)
                print(f"成功加载: {csv_file.name}, 共 {len(df)} 行数据")
            except Exception as e:
                print(f"读取 {csv_file.name} 失败: {e}")
                continue
        
        if len(doctor_dataframes) == 0:
            print("错误: 无法读取任何评分文件")
            return None
        
        # 计算三位医生的平均分并四舍五入取整
        print("计算三位医生的平均分...")
        
        # 使用第一个医生的数据作为基础结构
        result_df = doctor_dataframes[0].copy()
        
        # 需要计算平均值的列（跳过序号列）
        numeric_columns = result_df.columns[1:]  # 从第二列开始都是数值列
        
        for col in numeric_columns:
            if col in result_df.columns:
                # 收集所有医生在该列的数据
                col_values = []
                for df in doctor_dataframes:
                    if col in df.columns:
                        # 处理数值，忽略NaN和空值
                        values = pd.to_numeric(df[col], errors='coerce')
                        col_values.append(values)
                
                if col_values:
                    # 计算平均值
                    stacked_values = np.array(col_values)
                    # 使用nanmean忽略NaN值
                    avg_values = np.nanmean(stacked_values, axis=0)
                    # 四舍五入取整
                    result_df[col] = np.round(avg_values).astype(int)
        
        # 强制转换所有数值列为Python原生类型以避免JSON序列化问题
        for col in result_df.columns[1:]:  # 跳过序号列
            if col in result_df.columns:
                result_df[col] = result_df[col].apply(lambda x: int(x) if pd.notna(x) else None)
        
        print(f"成功计算平均分，共 {len(result_df)} 行数据")
        print("样本数据:")
        print(result_df.head(3))
        
        return result_df
