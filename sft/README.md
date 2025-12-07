---
library_name: peft
license: other
base_model: /data1/zhangyi/LargeModel/Meta-Llama-3-8B-Instruct
tags:
- base_model:adapter:/data1/zhangyi/LargeModel/Meta-Llama-3-8B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/data1/zhangyi/LargeModel/Meta-Llama-3-8B-Instruct](https://huggingface.co//data1/zhangyi/LargeModel/Meta-Llama-3-8B-Instruct) on the alpaca_gpt4_zh, the identity and the adgen_local datasets.
It achieves the following results on the evaluation set:
- Loss: 1.8040

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- num_epochs: 5.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 2.1221        | 0.4251 | 50   | 1.9806          |
| 1.9082        | 0.8502 | 100  | 1.8663          |
| 1.7489        | 1.2721 | 150  | 1.8246          |
| 1.7097        | 1.6971 | 200  | 1.8040          |
| 1.6743        | 2.1190 | 250  | 1.7967          |
| 1.5655        | 2.5441 | 300  | 1.7980          |
| 1.5738        | 2.9692 | 350  | 1.7870          |
| 1.4888        | 3.3911 | 400  | 1.8040          |
| 1.4382        | 3.8162 | 450  | 1.8056          |
| 1.4042        | 4.2380 | 500  | 1.8143          |
| 1.4189        | 4.6631 | 550  | 1.8180          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.9.1+cu128
- Datasets 4.0.0
- Tokenizers 0.22.1