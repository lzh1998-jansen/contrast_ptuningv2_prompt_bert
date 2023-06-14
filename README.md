# contrast_ptuningv2_prompt_bert
使用对比学习思路，通过p-tuningv2 finetune bert ，使用prompt bert 的思路，通过模板去构建句向量表征


1.预训练模型使用robert，可自行从huggingface下载，hfl/chinese-roberta-wwm-ext

2.修改一下预训练模型路径，创建 model_weight 文件夹

3.执行 python main.py，即刻完成train，infer
