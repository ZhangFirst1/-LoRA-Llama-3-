# app/config.py
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 路径配置
class Config:
    # 基础路径
    PROJECT_ROOT = PROJECT_ROOT
    
    # 模型路径
    BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "LLM-Research", "Meta-Llama-3-8B-Instruct")
    LORA_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "sft", "checkpoint-590")
    
    # 应用配置
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 7860
    
    # 模型参数
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_LENGTH = 1024
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录 - 现在只创建确实需要的目录"""
        # 检查并创建模型目录（如果不存在）
        if not os.path.exists(cls.BASE_MODEL_PATH):
            print(f"⚠️ 注意: 基础模型路径不存在: {cls.BASE_MODEL_PATH}")
        
        if not os.path.exists(cls.LORA_CHECKPOINT_PATH):
            print(f"⚠️ 注意: LoRA适配器路径不存在: {cls.LORA_CHECKPOINT_PATH}")
        
        print("✅ 配置加载完成")

# 初始化配置
Config.create_dirs()