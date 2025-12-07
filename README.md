# Attribute-Driven Ad Generation based on Llama-3 & LoRA

# 基于 LoRA 微调 Llama-3 的属性驱动型广告生成模型

---

## 1. 简介

本项目旨在解决电商行业内容生产中面临的成本高、效率低、质量参差不齐的痛点。

我们基于 **Meta Llama-3-8B-Instruct** 基座模型，利用 **LoRA参数高效微调技术**，构建了一个**属性驱动**  的广告文案生成模型。该模型能够接收结构化的商品属性列表（如类型、材质、风格），生成流畅、精准且极具营销吸引力的广告文案。

与通用大模型相比，本模型有效解决了“幻觉”问题和“风格平淡”问题，实现了从“说明书式文本”到“营销式文案”的风格迁移。

## 2. 核心功能

* **高效训练:** 采用 **4-bit 量化 (QLoRA)** 和 **梯度累积** 技术，成功在单张 **RTX 3090 (24GB)** 消费级显卡上完成了 8B 模型的微调。
* **显著提升:** 相比未微调的基座模型 (Zero-shot)，ROUGE-1 提升 **321%**，BLEU-4 提升 **623%**。
* **属性驱动:** 能够精准遵循输入的 Key-Value 属性约束，支持长尾属性（如特定材质、工艺）的精准生成。
* **开箱即用:** 提供基于 Gradio 的 WebUI 交互界面，支持流式生成。

---

## 3. 技术架构

本项目基于 PyTorch 和 Hugging Face 生态构建，核心技术栈包括：

* **Base Model:** Meta-Llama-3-8B-Instruct (GQA, RoPE, SwiGLU)
* **Fine-tuning:** LoRA (Rank=8, Alpha=16, Target Modules=`q_proj,v_proj`)
* **Optimization:** Bitsandbytes (NF4 Quantization), Gradient Accumulation
* **Dataset:** Tsinghua AdGen (Advertising Generation) Benchmark

---

## 4. 实验结果

我们在 AdGen 测试集上进行了定量评估，并与原始 Llama-3 模型进行了对比。

### 1. 定量评估

| 模型 (Model)        | ROUGE-1   | ROUGE-2   | ROUGE-L   | BLEU-4    | 提升幅度         |
| :------------------ | :-------- | :-------- | :-------- | :-------- | :--------------- |
| Llama-3 (Zero-shot) | 10.16     | 2.08      | 6.49      | 2.77      | -                |
| **Ours (LoRA)**     | **42.77** | **21.66** | **35.20** | **20.03** | **+623% (BLEU)** |

### 2. 定性分析

| 属性输入 (Input)                                | Llama-3 (Zero-shot)                                        | **Ours (Fine-tuned)**                                        |
| :---------------------------------------------- | :--------------------------------------------------------- | :----------------------------------------------------------- |
| `类型#口红 * 质地#丝绒 * 功效#显白 * 场景#约会` | “这是一款丝绒质地的口红，颜色很显白，非常适合约会时使用。” | “一抹丝绒哑光，高级感扑面而来。超级显白的色调，让你在约会时刻气场全开，轻松俘获他的心！” |

---

## 5. 快速开始

### 环境要求

* Python 3.10+
* PyTorch 2.1.2+
* CUDA 12.1+

### 1. 安装依赖

```bash
git clone [https://github.com/ZhangFirst1/-LoRA-Llama-3-.git](https://github.com/ZhangFirst1/-LoRA-Llama-3-.git)
cd -LoRA-Llama-3-
pip install -r requirements.txt
```

### 2. 数据准备

本项目使用 [AdGen 数据集](https://huggingface.co/datasets/HasturOfficial/adgen)。请下载数据并放置在 `data/` 目录下，并使用 `src/process_data.py` 转换为 Alpaca/ShareGPT 格式。

```
python src/process_data.py --raw_file ./data/raw_data.json
```

### 3. 启动训练

我们提供了一键启动脚本。请确保您的显存大于 22GB。

```bash
bash scripts/run_sft.sh
```

核心训练参数配置：

- `learning_rate`: 2e-4 (Cosine scheduler)
- `lora_rank`: 8
- `batch_size`: 2 (per device)
- `gradient_accumulation_steps`: 4
- `quantization_bit`: 4

### 4. 启动推理 Demo

训练完成后，运行以下命令启动可视化界面：

Bash

```
python src/main.py
```

------

## 6. 项目结构

```Plaintext
-LoRA-Llama-3-/
├── data/                      # 数据目录
├── model/                     # 模型目录
├── results/                   # 结果目录
│   ├── train_loss.png
│   └── eval_results.json
├── scripts/                   # 脚本目录
│   └── run_sft.sh             # 核心启动脚本
├── src/                       # 源码目录
│   ├── train.py               # 训练代码
│   ├── evaluate.py            # 测试与评估代码
│   ├── main.py            	   # 可视化界面
│   └── chat_model.py          # 加载模型
│   └── process_data.py        # 数据预处理
├── README.md                  # 项目说明文档
└── requirements.txt           # 依赖包列表
```
