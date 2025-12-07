# src/evaluate.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import json

def evaluate():
    # 路径配置
    base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "./model/lora_adapter"
    test_data_path = "./data/adgen_dev.json"

    # 加载模型
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 加载测试数据
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)[:50] # 仅测试前50条用于演示

    rouge = Rouge()
    scores = {'rouge-1': [], 'rouge-l': [], 'bleu-4': []}

    print("Starting evaluation...")
    for item in test_data:
        input_text = item['input']
        reference = item['output']
        
        # 构造 Prompt
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # 推理
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()

      
        try:
            r_score = rouge.get_scores(" ".join(generated_text), " ".join(reference))
            scores['rouge-1'].append(r_score[0]['rouge-1']['f'])
            scores['rouge-l'].append(r_score[0]['rouge-l']['f'])
        except:
            pass # 防止空字符串报错

    # 计算平均分
    avg_rouge1 = sum(scores['rouge-1']) / len(scores['rouge-1'])
    print(f"Evaluation Result: ROUGE-1: {avg_rouge1:.4f}")
    
    # 保存结果
    with open("./results/eval_scores.json", "w") as f:
        json.dump({"ROUGE-1": avg_rouge1}, f)

if __name__ == "__main__":
    evaluate()