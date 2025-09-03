"""
批量处理器模块
负责批量处理多个视频文件
"""

from pathlib import Path
import pandas as pd
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
        """加载评分表格"""
        # 查找xlsx或xls文件
        score_files = []
        for ext in ['*.xlsx', '*.xls']:
            score_files.extend(input_path.glob(ext))
        
        # 过滤掉临时文件
        score_files = [f for f in score_files if not f.name.startswith('~$')]
        
        if not score_files:
            print("警告: 未找到评分表格文件")
            return None
        
        score_file = score_files[0]
        print(f"加载评分表格: {score_file.name}")
        
        try:
            df = pd.read_excel(score_file, header=0)
            print(f"成功加载评分表格，共 {len(df)} 行数据")
            return df
        except Exception as e:
            print(f"读取评分表格失败: {e}")
            try:
                df = pd.read_excel(score_file, header=None)
                print(f"成功加载评分表格(无标题)，共 {len(df)} 行数据")
                return df
            except Exception as e2:
                print(f"读取评分表格失败: {e2}")
                return None
