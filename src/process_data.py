import json
import os
import random
import argparse
from tqdm import tqdm

# 配置随机种子，保证复现性
random.seed(42)

def parse_adgen_content(content_str):
    """
    解析 AdGen 的 content 字段
    输入示例: "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤"
    输出示例: "类型: 裤; 版型: 宽松; 风格: 性感; 图案: 线条; 裤型: 阔腿裤"
    """
    if not content_str:
        return ""
    
    # 1. 替换分隔符
    # AdGen 使用 '*' 分隔属性，'#' 分隔键值
    properties = content_str.split('*')
    parsed_props = []
    
    for prop in properties:
        if '#' in prop:
            try:
                key, value = prop.split('#', 1)
                parsed_props.append(f"{key}: {value}")
            except ValueError:
                continue
                
    return "; ".join(parsed_props)

def format_data(raw_file_path, output_dir, split_ratio=0.99):
    """
    读取原始数据，清洗、格式化并划分数据集
    """
    data_list = []
    
    print(f"🔄 正在读取原始数据: {raw_file_path} ...")
    
    # 读取原始数据 (假设原始数据是 json 格式，或者是每行一个 json)
    # 这里兼容每行一个 JSON 对象的格式 (JSONL)
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    
    # 定义 System Prompt
    system_prompt = "你是一个专业的电商文案策划师，请根据以下商品属性，撰写一段吸引人的营销文案。"
    
    # 统计变量
    total_tokens = 0
    max_len = 0
    skipped_count = 0
    
    for line in tqdm(lines, desc="Processing"):
        try:
            item = json.loads(line.strip())
            
            raw_content = item.get('content', '')
            summary = item.get('summary', '')
            
            # --- 数据清洗逻辑 ---
            # 1. 过滤掉 summary 太短的样本 (可能是脏数据)
            if len(summary) < 10:
                skipped_count += 1
                continue
                
            # 2. 解析 Input
            parsed_input = parse_adgen_content(raw_content)
            
            if not parsed_input:
                skipped_count += 1
                continue

            # 3. 简单的文本清洗 (去除可能的 HTML 标签或乱码)
            summary = summary.replace("&nbsp;", " ").strip()

            # --- 构建 Alpaca 格式 ---
            entry = {
                "instruction": system_prompt,
                "input": parsed_input,
                "output": summary
            }
            
            # 简单的长度统计 (按字符估算)
            cur_len = len(parsed_input) + len(summary)
            total_tokens += cur_len
            if cur_len > max_len:
                max_len = cur_len
                
            data_list.append(entry)
            
        except json.JSONDecodeError:
            continue

   
    
    # --- 数据集划分 ---
    random.shuffle(data_list)
    split_idx = int(len(data_list) * split_ratio)
    
    train_data = data_list[:split_idx]
    dev_data = data_list[split_idx:]
    
    # --- 确保输出目录存在 ---
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 保存文件 ---
    train_path = os.path.join(output_dir, "adgen_train.json")
    dev_path = os.path.join(output_dir, "adgen_dev.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open(dev_path, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=2)
        

    
    # 生成 dataset_info.json (LLaMA-Factory 需要，虽然你现在用自定义脚本，但保留这个是个好习惯)
    dataset_info = {
        "adgen_train": {
            "file_name": "adgen_train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        },
        "adgen_dev": {
            "file_name": "adgen_dev.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }
    with open(os.path.join(output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdGen 数据预处理脚本")
    parser.add_argument("--raw_file", type=str, default="./data/raw_data.json", help="原始 AdGen 数据文件路径 (JSONL格式)")
    parser.add_argument("--output_dir", type=str, default="./data", help="处理后数据的保存目录")
    
    args = parser.parse_args()
    
    # 检查原始文件是否存在
    if not os.path.exists(args.raw_file):
        print(f"❌ 错误: 找不到原始文件 {args.raw_file}")
        print("请下载 AdGen 数据集 (train.json) 并放置在 data 目录下，或使用 --raw_file 指定路径。")
    else:
        format_data(args.raw_file, args.output_dir)