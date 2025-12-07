# app/chat_model.py
import re
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging
import os
from app.config import Config

logger = logging.getLogger(__name__)

class LoraChatModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.config = Config
    
    def load_model(self):
        """加载模型"""
        try:
            logger.info("正在加载Meta-Llama-3-8B-Instruct模型...")
            
            base_model_path = self.config.BASE_MODEL_PATH
            lora_path = self.config.LORA_CHECKPOINT_PATH
            
            if not os.path.exists(base_model_path):
                raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
            
            if not os.path.exists(lora_path):
                logger.warning(f"LoRA适配器路径不存在: {lora_path}，将使用原始模型")
                use_lora = False
            else:
                use_lora = True
                logger.info("✅ 找到LoRA适配器")
            
            # 1. 加载分词器
            logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 2. 加载基础模型
            logger.info("加载基础模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            # 3. 加载LoRA适配器
            if use_lora:
                try:
                    logger.info("加载LoRA适配器...")
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    logger.info("✅ LoRA适配器加载成功")
                except Exception as e:
                    logger.warning(f"LoRA适配器加载失败，使用原始模型: {e}")
                    use_lora = False
            
            self.is_loaded = True
            model_type = "LoRA微调版" if use_lora else "原始模型"
            logger.info(f"✅ Meta-Llama-3-8B-Instruct {model_type}加载成功！")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def chat(self, message, history=None, temperature=0.7, max_length=1024):
        """生成回复 - 修复对话历史处理问题"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            # 修复：确保每个问题独立处理
            clean_history = self._validate_and_clean_history(history or [])
            
            # 构建正确的对话提示
            prompt = self._build_correct_prompt(message, clean_history)
            
            # 将输入移到GPU
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=min(max_length, 500),
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码回复
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 彻底清理回复内容 - 确保只返回当前问题的回答
            clean_response = self._extract_clean_response_for_current_question(response, prompt, message)
            
            return clean_response
            
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return f"抱歉，生成回复时出现错误: {str(e)}"
    
    def _validate_and_clean_history(self, history):
        """验证和清理历史记录，防止问题累积"""
        clean_history = []
        
        # 确保历史记录格式正确
        for i, item in enumerate(history):
            if isinstance(item, tuple) and len(item) == 2:
                # 清理历史中的assistant前缀
                user_msg, assistant_msg = item
                clean_assistant_msg = self._remove_assistant_prefix(assistant_msg)
                clean_history.append((user_msg, clean_assistant_msg))
            elif isinstance(item, dict) and 'role' in item and 'content' in item:
                # 转换字典格式
                if item['role'] == 'user':
                    current_user_msg = item['content']
                elif item['role'] == 'assistant':
                    clean_assistant_msg = self._remove_assistant_prefix(item['content'])
                    clean_history.append((current_user_msg, clean_assistant_msg))
        
        # 限制历史记录长度，防止累积过多上下文
        if len(clean_history) > 5:  # 最多保留5轮对话
            clean_history = clean_history[-5:]
        
        return clean_history
    
    def _build_correct_prompt(self, current_message, history):
        """构建正确的提示词，明确指示回复格式"""
        system_prompt = """<|start_header_id|>system<|end_header_id|>\n\n
你是一个有帮助的AI助手。请针对用户的最新问题进行直接回答，不要以"Assistant"、"助手"或任何类似前缀开头，直接给出答案内容。<|eot_id|>\n"""
        
        conversation = system_prompt
        
        # 添加历史对话（如果有）
        for i, (user_msg, assistant_msg) in enumerate(history):
            conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>\n"
            conversation += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>\n"
        
        # 添加当前问题
        conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{current_message}<|eot_id|>\n"
        conversation += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return conversation
    
    def _extract_clean_response_for_current_question(self, full_response, prompt, current_question):
        """专门为当前问题提取干净的回复"""
        # 确保只提取当前问题的回答
        if full_response.startswith(prompt):
            response_content = full_response[len(prompt):].strip()
        else:
            # 如果格式不匹配，尝试其他提取方法
            response_content = self._extract_by_current_question(full_response, current_question)
        
        # 彻底清理回复 - 特别加强assistant开头的清理
        clean_content = self._aggressive_clean_response(response_content)
        
        # 新增：专门处理以assistant开头的情况
        clean_content = self._remove_assistant_prefix(clean_content)
        
        # 验证回复是否针对当前问题
        if not self._is_response_relevant(clean_content, current_question):
            logger.warning(f"回复可能不相关。问题: {current_question}, 回复: {clean_content}")
        
        return clean_content
    
    def _remove_assistant_prefix(self, text):
        """专门移除assistant相关前缀"""
        if not text:
            return text
        
        # 定义需要移除的assistant前缀模式
        assistant_prefixes = [
            r'^(assistant|Assistant|ASSISTANT)[:\s]*',
            r'^(助手|AI助手|AI助理|机器人)[：:\s]*',
            r'^(好的|明白了|根据您的问题|针对这个问题|关于这个问题)[，,\s]*',
            r'^[Aa]ssistant\s+is\s+',
            r'^[Aa]ssistant\s*[：:]\s*'
        ]
        
        for prefix in assistant_prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_by_current_question(self, text, current_question):
        """根据当前问题特征提取回复"""
        # 查找最后一个assistant标记后的内容
        markers = [
            "<|start_header_id|>assistant<|end_header_id|>",
            "assistant:",
            "Assistant:",
            "助手:",
            "助手："
        ]
        
        for marker in markers:
            if marker in text:
                parts = text.split(marker)
                if len(parts) > 1:
                    # 取最后一个assistant标记后的内容
                    content = parts[-1].strip()
                    # 去除所有特殊标记
                    content = re.sub(r'<\|.*?\|>', '', content)
                    # 移除assistant前缀
                    content = self._remove_assistant_prefix(content)
                    return content
        
        # 如果找不到标记，尝试基于问题内容提取
        if current_question in text:
            parts = text.split(current_question)
            if len(parts) > 1:
                content = parts[-1].strip()
                content = self._remove_assistant_prefix(content)
                return content
        
        # 最终清理
        content = text.strip()
        content = self._remove_assistant_prefix(content)
        return content
    
    def _aggressive_clean_response(self, text):
        """加强版的清理回复内容"""
        if not text:
            return text
        
        # 首先移除assistant相关前缀
        text = self._remove_assistant_prefix(text)
        
        # 去除所有特殊标记
        patterns_to_remove = [
            r'<\|start_header_id\|>.*?<\|end_header_id\|>',
            r'<\|eot_id\|>',
            r'<\|.*?\|>',
            r'\[.*?\]',
            r'\(.*?\)'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # 去除常见的不必要前缀
        prefixes_to_remove = [
            r'^(好的|根据|针对|关于|这件|这款|这是一个|首先|那么|另外)[，,]\s*',
            r'^[\"\'「」【】\[\]\(\)]\s*',
            r'^(嗯|啊|呃|那个|这个)[，,]\s*',
            r'^(那么|接下来|此外|另外)[，,]\s*'
        ]
        
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)
        
        # 清理多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()
        
        # 确保不以标点符号开头
        if text and text[0] in '，,.:：；;!！?？':
            text = text[1:].strip()
        
        # 如果文本以"这款"开头，但前面没有内容，直接保留
        if text.startswith('这款'):
            # 检查是否只是单纯的"这款"开头，如果是商品描述，可以保留
            pass
        
        return text
    
    def _is_response_relevant(self, response, question):
        """检查回复是否与当前问题相关"""
        if not response or not question:
            return False
        
        # 简单的关键词匹配检查
        question_keywords = self._extract_keywords(question)
        response_keywords = self._extract_keywords(response)
        
        # 如果有共同关键词，认为相关
        common_keywords = set(question_keywords) & set(response_keywords)
        return len(common_keywords) > 0 or len(response) > 10  # 或者回复长度大于10也认为相关
    
    def _extract_keywords(self, text):
        """提取文本中的关键词"""
        if not text:
            return []
        
        # 简单的关键词提取
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text)
        # 过滤掉常见虚词
        stop_words = {'的', '了', '是', '在', '有', '和', '就', '都', '而', '及', '与', '这', '那', '你', '我', '他', '她', '它'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords