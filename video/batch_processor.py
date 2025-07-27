"""
批量处理器模块
负责批量处理多个视频文件
"""

from pathlib import Path
from typing import Set
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
          # 查找所有视频文件（避免重复）
        video_files = set()  # 使用set避免重复
        for ext in self.video_extensions:
            # 同时递归搜索小写和大写扩展名
            video_files.update(input_path.glob(f"**/*{ext}"))
            video_files.update(input_path.glob(f"**/*{ext.upper()}"))
        
        video_files = list(video_files)  # 转换回list
        
        if not video_files:
            print(f"在目录 {input_dir} 中未找到视频文件")
            return
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        # 处理每个视频
        success_count = 0
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] 开始处理: {video_file.name}")
            
            if self.video_processor.process_single_video(str(video_file), output_dir):
                success_count += 1
        
        print(f"\n批量处理完成！")
        print(f"成功处理: {success_count}/{len(video_files)} 个视频")
