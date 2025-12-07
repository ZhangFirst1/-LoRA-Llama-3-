# app/main.py
import gradio as gr
import logging
import os
from app.chat_model import LoraChatModel
from app.config import Config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_chat_interface():
    """创建优化布局的聊天界面"""
    
    # 初始化模型
    chat_model = LoraChatModel()
    
    def respond(message, chat_history, temperature, max_length):
        try:
            if not message.strip():
                return "", chat_history, "就绪"
            
            # 转换历史记录格式
            history_for_model = []
            for msg in chat_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    if msg['role'] == 'user':
                        human_msg = msg['content']
                    elif msg['role'] == 'assistant':
                        history_for_model.append((human_msg, msg['content']))
            
            # 生成回复
            response = chat_model.chat(
                message=message,
                history=history_for_model,
                temperature=temperature,
                max_length=max_length
            )
            
            # 更新历史记录
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            
            return "", chat_history, "回复生成成功"
            
        except Exception as e:
            error_msg = f"生成失败: {str(e)}"
            logger.error(error_msg)
            return "", chat_history, f"❌ {error_msg}"
    
    def clear_chat():
        return [], "对话已清空"
    
    def initialize_model():
        """初始化模型"""
        try:
            chat_model.load_model()
            return "模型就绪"
        except Exception as e:
            return f"❌ 加载失败: {str(e)}"
    
    # 创建优化布局的界面
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        title="广告生成助手",
        css="""
        /* 整体布局优化 - 缩小尺寸 */
        .gradio-container {
            max-width: 1200px !important;
            width: 95% !important;
            margin: 0 auto !important;
            padding: 10px !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa !important;
            min-height: 100vh !important;
            overflow: auto !important;
        }
        
        /* 删除问题示例的边框 */
        div.svelte-1nguped {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* 主布局 - 缩小整体高度 */
        .main-layout {
            gap: 15px;
            align-items: stretch;
            height: auto !important;
            min-height: 600px !important;
            overflow: visible !important;
        }
        
        /* 侧边栏 - 缩小宽度 */
        .sidebar-column {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            height: auto !important;
            width: 250px !important;
            min-width: 200px !important;
            overflow: visible !important;
        }
        
        /* 主聊天区域 - 缩小宽度 */
        .chat-column {
            display: flex;
            flex-direction: column;
            gap: 10px;
            height: auto !important;
            flex: 1;
            min-width: 600px !important;
            overflow: visible !important;
        }
        
        /* 聊天区域 - 显著缩小高度 */
        .chatbot-area {
            flex: 1;
            min-height: 300px !important;
            height: 300px !important;
            background: white !important;
            border-radius: 8px !important;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow-y: auto !important;
            width: 100% !important;
            border: 1px solid #e0e0e0 !important;
            margin-bottom: 10px !important;
        }
        
        /* 隐藏聊天框右下角的处理状态 */
        .chatbot-area .svelte-1w6e6tj, /* 处理状态容器 */
        .chatbot-area .svelte-1w6e6tj::after, /* 处理状态伪元素 */
        .chatbot-area [data-testid="bot-status"], /* 机器人状态 */
        .chatbot-area .generating { /* 生成中状态 */
            display: none !important;
            visibility: hidden !important;
        }
        
        /* 隐藏打字机效果 */
        .chatbot-area .typing {
            display: none !important;
        }
        
        /* 输入区域 - 确保显示 */
        .input-container {
            position: relative;
            background: white !important;
            border-radius: 8px !important;
            padding: 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 0;
            width: 100% !important;
            border: 1px solid #e0e0e0 !important;
            height: auto !important;
            display: block !important;
        }
        
        /* 输入框样式 - 缩小高度 */
        .input-with-button {
            position: relative;
            width: 100%;
            display: block !important;
        }
        
        .input-with-button textarea {
            width: 100% !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 50px 10px 15px !important;
            resize: vertical !important;
            min-height: 60px !important;
            max-height: 80px !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
            background: white !important;
            overflow: auto !important;
            display: block !important;
        }
        
        .input-with-button textarea:focus {
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
            background: white !important;
        }
        
        /* 发送按钮样式 - 调整位置 */
        .send-icon-button {
            position: absolute !important;
            right: 10px !important;
            bottom: 10px !important;
            width: 35px !important;
            height: 35px !important;
            border-radius: 50% !important;
            background: #667eea !important;
            border: none !important;
            color: white !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            z-index: 1000 !important;
            font-size: 12px !important;
        }
        
        .send-icon-button:hover {
            background: #5a6fd8 !important;
            transform: scale(1.05) !important;
        }
        
        /* 示例提示词区域 - 确保显示 */
        .prompt-examples-area {
            background: white;
            padding: 10px 15px !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 0;
            width: 100% !important;
            border: 1px solid #e0e0e0 !important;
            margin-top: 10px !important;
            height: auto !important;
            min-height: auto !important;
            display: block !important;
        }
        
        /* 参数组样式 */
        .param-group {
            margin-bottom: 15px !important;
        }
        
        .param-group .gradio-slider {
            margin: 8px 0 !important;
        }
        
        /* 滑块样式 */
        .gradio-slider .range {
            background: #667eea !important;
            height: 4px !important;
        }
        
        .gradio-slider .thumb {
            border: 2px solid #667eea !important;
            width: 16px !important;
            height: 16px !important;
        }
        
        /* 操作按钮 */
        .action-buttons {
            margin: 10px 0 !important;
        }
        
        .clear-button {
            background: #6c757d !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            font-size: 12px !important;
        }
        
        .clear-button:hover {
            background: #5a6268 !important;
        }
        
        /* 状态显示 */
        .status-box {
            background: #f8f9fa;
            padding: 8px !important;
            border-radius: 6px;
            margin-top: 10px !important;
            border-left: 3px solid #667eea !important;
            font-size: 11px !important;
            height: auto !important;
        }
        
        /* 标题样式 */
        h1 {
            margin-bottom: 10px !important;
            color: #2c3e50 !important;
            font-size: 1.3em !important;
            font-weight: 600 !important;
            text-align: center !important;
        }
        
        h2, h3 {
            margin-bottom: 8px !important;
            color: #2c3e50 !important;
            font-weight: 600 !important;
            font-size: 0.9em !important;
        }
        
        /* 示例提示词按钮样式 */
        .gradio-examples {
            border: none !important;
            background: transparent !important;
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 5px !important;
        }
        
        .gradio-example {
            background: #667eea !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 4px 8px !important;
            margin: 0 !important;
            transition: all 0.2s ease !important;
            font-size: 10px !important;
            flex: 1;
            min-width: 80px;
            text-align: center;
        }
        
        .gradio-example:hover {
            background: #5a6fd8 !important;
        }
        
        /* 聊天消息样式 */
        .user-message {
            background: #667eea !important;
            color: white !important;
            border: none !important;
            border-radius: 8px 8px 2px 8px !important;
            margin: 4px 0 !important;
            padding: 8px 12px !important;
            max-width: 80% !important;
            margin-left: auto !important;
            font-size: 12px !important;
        }
        
        .bot-message {
            background: #f8f9fa !important;
            color: #2c3e50 !important;
            border: none !important;
            border-radius: 8px 8px 8px 2px !important;
            margin: 4px 0 !important;
            padding: 8px 12px !important;
            max-width: 80% !important;
            font-size: 12px !important;
        }
        
        /* 输入框占位符样式 */
        .gradio-textbox textarea::placeholder {
            color: #999 !important;
            font-size: 13px !important;
        }
        
        /* 响应式设计 */
        @media (max-width: 1200px) {
            .gradio-container {
                max-width: 98% !important;
                width: 98% !important;
                padding: 5px !important;
            }
            
            .main-layout {
                flex-direction: column !important;
                height: auto !important;
            }
            
            .sidebar-column {
                width: 100% !important;
                margin-bottom: 10px !important;
            }
            
            .chat-column {
                width: 100% !important;
                min-width: auto !important;
            }
        }
        
        /* 滚动条 */
        .chatbot-area::-webkit-scrollbar {
            width: 4px;
        }
        
        .chatbot-area::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        .chatbot-area::-webkit-scrollbar-thumb {
            background: #667eea;
        }
        
        /* 标签样式 - 确保显示 */
        .gradio-label {
            font-weight: 600 !important;
            color: #2c3e50 !important;
            margin-bottom: 4px !important;
            font-size: 12px !important;
            display: block !important;
        }
        
        /* 确保所有元素都显示 */
        .gradio-chatbot, .gradio-textbox, .gradio-group {
            display: block !important;
            visibility: visible !important;
        }
        
        /* 强制显示所有隐藏元素 */
        [class*="gradio"] {
            display: block !important;
            visibility: visible !important;
        }
        """
    ) as demo:
        
        gr.Markdown("# 广告生成助手")
        
        with gr.Row(elem_classes="main-layout"):
            # 左侧参数区域
            with gr.Column(scale=1, min_width=250, elem_classes="sidebar-column"):
                gr.Markdown("### 参数设置")
                
                # 参数组
                with gr.Group(elem_classes="param-group"):
                    temperature = gr.Slider(
                        0.1, 1.5, 
                        value=0.7,
                        label="创造性",
                        info="值越高，回复越有创意"
                    )
                    
                    max_length = gr.Slider(
                        100, 2048, 
                        value=1024,
                        step=100, 
                        label="回复长度",
                        info="控制回复的最大长度"
                    )
                
                # 操作按钮
                with gr.Row(elem_classes="action-buttons"):
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes="clear-button")
                
                # 状态显示
                with gr.Group(elem_classes="status-box"):
                    status = gr.Textbox(
                        label="系统状态",
                        value="正在初始化模型...",
                        interactive=False,
                        lines=2
                    )
            
            # 右侧聊天区域
            with gr.Column(scale=3, min_width=600, elem_classes="chat-column"):
                # 聊天显示区域
                with gr.Group(elem_classes="chatbot-area"):
                    chatbot = gr.Chatbot(
                        type='messages',
                        label="对话记录",
                        show_copy_button=True,
                        container=False,
                        height=300
                    )
                
                # 输入区域
                with gr.Group(elem_classes="input-container"):
                    with gr.Group(elem_classes="input-with-button"):
                        msg = gr.Textbox(
                            label="请输入您的问题",
                            placeholder="请输入您的问题...（按Enter发送，Shift+Enter换行）",
                            lines=2,
                            container=False,
                            show_label=True
                        )
                        # 发送图标按钮
                        send_btn = gr.Button(
                            "➤",
                            variant="primary", 
                            elem_classes="send-icon-button"
                        )
                
                # 示例提示词区域
                with gr.Group(elem_classes="prompt-examples-area"):
                    gr.Markdown("#### 示例提示词")
                    gr.Examples(
                        examples=[
                            "类型#上衣*材质#牛仔裤*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞",
                            "类型#连衣裙*材质#雪纺*风格#正式*衣型*颜色#淡紫色",
                            "类型#耳机*特点#降噪*适用#游戏",
                            "类型#口红*质地#丝绒*功能#易启*场景#约会"
                        ],
                        inputs=msg,
                        label="点击快速输入",
                        examples_per_page=4
                    )
        
        
        # 事件绑定
        msg.submit(respond, [msg, chatbot, temperature, max_length], [msg, chatbot, status])
        send_btn.click(respond, [msg, chatbot, temperature, max_length], [msg, chatbot, status])
        clear_btn.click(clear_chat, outputs=[chatbot, status])
        
        # 页面加载时初始化模型
        demo.load(initialize_model, outputs=[status])
    
    return demo

if __name__ == "__main__":
    print("启动广告生成助手...")
    print(f"服务地址: http://{Config.SERVER_HOST}:{Config.SERVER_PORT}")
    
    # 创建并启动界面
    demo = create_chat_interface()
    demo.launch(
        server_name=Config.SERVER_HOST,
        server_port=Config.SERVER_PORT,
        share=False
    )