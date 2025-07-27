"""
面部表情分析系统 - 主入口文件
整合所有模块，提供统一的入口点
"""
import matplotlib
matplotlib.use('Agg')

import logging
from video.batch_processor import BatchProcessor
from config import Config


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL), 
                       format=Config.LOG_FORMAT)
    
    print("面部表情分析系统（模块化版本）")
    print("批量处理文件夹中的所有视频")
    print("-" * 50)
    print(f"模型路径: {Config.MODEL_PATH}")
    print(f"输入目录: {Config.INPUT_DIR}")
    print(f"输出目录: {Config.OUTPUT_DIR}")
    print("-" * 50)
    
    try:
        # 验证配置
        Config.validate_paths()
        
        # 创建批量处理器并运行
        batch_processor = BatchProcessor(Config.MODEL_PATH)
        batch_processor.process_directory(Config.INPUT_DIR, Config.OUTPUT_DIR)
        
    except Exception as e:
        logging.error(f"程序运行出错: {e}")
        print(f"错误: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()
