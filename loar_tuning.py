from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForMaskedLM
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from data_process import load_data
from model import compute_loss
from sklearn.metrics import accuracy_score


def seed_everything(seed=3427):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


"""
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
"""


def init_tokenizer(path):
    # [X]
    tokenizer = AutoTokenizer.from_pretrained(path)

    tokenizer.add_tokens(['[X]'])
    return tokenizer


class PromptBERT(nn.Module):
    def __init__(self, pretrained_model_path, dropout_prob, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        # 设置dropout_prob 可以提高分数
        conf = AutoConfig.from_pretrained(pretrained_model_path)

        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob

        self.bert = AutoModel.from_pretrained(pretrained_model_path, config=conf)

        self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        #### 切记已经加入这个【x】这个token  2w _>20001
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, prompt_input_ids, prompt_attention_mask, template_input_ids, template_attention_mask):
        prompt_mask_emb = self.cal_mask_embedding(prompt_input_ids, prompt_attention_mask)
        template_mask_emb = self.cal_mask_embedding(template_input_ids, template_attention_mask)
        return prompt_mask_emb - template_mask_emb

    def cal_mask_embedding(self, input_ids, attention_mask):
        # [b,s,dim]
        last_hidden, _ = self.bert(input_ids, attention_mask, return_dict=False)
        # 我爱你 这句话的意思为 【mask】 -> 101 p,p,p,pp,p,p,p,p ,103,102   ->[false,false,....true,false]
        # [b,s]
        mask_index = (input_ids == self.mask_id).long()
        # [b,s,dim]
        # [b,s]->[b,s,1]->[b,s,dim]
        input_mask_expand = mask_index.unsqueeze(-1).expand(last_hidden.size()).float()

        # [b,s,dim]->[b,dim]
        mask_emb = torch.sum(last_hidden * input_mask_expand, dim=1)
        return mask_emb


class Lora_Model(nn.Module):
    def __init__(self, pretrained_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = AutoModel.from_pretrained(pretrained_path)
        self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        #### 切记已经加入这个【x】这个token  2w _>20001
        self.bert.resize_token_embeddings(len(self.tokenizer))
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.lora_model = get_peft_model(self.bert, peft_config)
        self.lora_model.gradient_checkpointing_enable()
        self.lora_model.enable_input_require_grads()
        self.lora_model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )
        self.lora_model.print_trainable_parameters()

    def forward(self,prompt_input_ids, prompt_attention_mask,  template_input_ids,template_attention_mask):
        prompt_mask_emb = self.cal_mask_embedding(prompt_input_ids, prompt_attention_mask)
        template_mask_emb = self.cal_mask_embedding(template_input_ids, template_attention_mask)
        return prompt_mask_emb - template_mask_emb

    def cal_mask_embedding(self, input_ids, attention_mask):
        # [b,s,dim]
        last_hidden, _ = self.bert(input_ids, attention_mask, return_dict=False)
        # 我爱你 这句话的意思为 【mask】 -> 101 p,p,p,pp,p,p,p,p ,103,102   ->[false,false,....true,false]
        # [b,s]
        mask_index = (input_ids == self.mask_id).long()
        # [b,s,dim]
        # [b,s]->[b,s,1]->[b,s,dim]
        input_mask_expand = mask_index.unsqueeze(-1).expand(last_hidden.size()).float()

        # [b,s,dim]->[b,dim]
        mask_emb = torch.sum(last_hidden * input_mask_expand, dim=1)
        return mask_emb


def train():
    best_acc = 0
    for e in range(epoch):
        print(f"当前是第{e}轮")
        lora.train()
        pbar = tqdm(train_data_loader)
        for data in pbar:
            optimzer.zero_grad()
            input_ids_prompt_sen_1 = data['input_ids_prompt_sen_1'].to(device)
            attention_masks_prompt_sen_1 = data['attention_masks_prompt_sen_1'].to(device)

            input_ids_template_sen_1 = data['input_ids_template_sen_1'].to(device)
            attention_masks_template_sen_1 = data['attention_masks_template_sen_1'].to(device)

            input_ids_prompt_sen_2 = data['input_ids_prompt_sen_2'].to(device)
            attention_masks_prompt_sen_2 = data['attention_masks_prompt_sen_2'].to(device)

            input_ids_template_sen_2 = data['input_ids_template_sen_2'].to(device)
            attention_masks_template_sen_2 = data['attention_masks_template_sen_2'].to(device)

            # h_i-h_i^
            s1 = lora.forward(input_ids_prompt_sen_1, attention_masks_prompt_sen_1,
                                          input_ids_template_sen_1,
                                          attention_masks_template_sen_1)
            # h_i'-h_i'^
            s2 = lora.forward(input_ids_prompt_sen_2, attention_masks_prompt_sen_2,
                                          input_ids_template_sen_2,
                                          attention_masks_template_sen_2)
            loss = compute_loss(s1, s2)
            loss.backward()
            optimzer.step()
            pbar.update()
            pbar.set_description(f"当前loss={loss.item():.6f}")

        pred = []
        label = []
        lora.eval()
        for data in tqdm(dev_data_loader):
            input_ids_prompt_sen_1 = data["input_ids_prompt_sen_1"].to(device)
            attention_masks_prompt_sen_1 = data["attention_masks_prompt_sen_1"].to(device)
            input_ids_template_sen_1 = data["input_ids_template_sen_1"].to(device)
            attention_masks_template_sen_1 = data["attention_masks_template_sen_1"].to(device)

            input_ids_prompt_sen_2 = data["input_ids_prompt_sen_2"].to(device)
            attention_masks_prompt_sen_2 = data["attention_masks_prompt_sen_2"].to(device)
            input_ids_template_sen_2 = data["input_ids_template_sen_2"].to(device)
            attention_masks_template_sen_2 = data["attention_masks_template_sen_2"].to(device)

            labels = data["label"].to(device)

            #  ,sim,sentence_embedding
            # 前向传播

            with torch.no_grad():
                # 前向传播
                sentence_embedding_a = lora.forward(input_ids_prompt_sen_1, attention_masks_prompt_sen_1,
                                                                input_ids_template_sen_1,
                                                                attention_masks_template_sen_1)
                sentence_embedding_b = lora.forward(input_ids_prompt_sen_2, attention_masks_prompt_sen_2,
                                                                input_ids_template_sen_2,
                                                                attention_masks_template_sen_2)

            similarity = F.cosine_similarity(sentence_embedding_a, sentence_embedding_b, dim=-1)
            # 获取预测值
            similarity = similarity.detach().cpu().numpy()
            pred.extend(similarity)
            # 获取标签
            label.extend(labels.cpu().numpy())
        # 计算验证集准确率
        pred = torch.tensor(pred)
        pred = (pred >= 0.7).long().detach().cpu().numpy()
        acc = accuracy_score(pred, label)
        print('dev acc:', acc)
        print()
        # 如果当前准确率大于最佳准确率，则保存模型参数
        if acc > best_acc:
            torch.save(lora.state_dict(), './model_weight/pt2_prompt.bin')
            best_acc = acc


def infer(thres_hold=0.5):
    # t1->emb->cos
    print(f"prompt bert 开始推理")

    lora.load_state_dict(torch.load('./model_weight/pt2_prompt.bin', map_location=device))
    lora.eval()

    pre = []
    label = []
    for data in dev_data_loader:
        input_ids_prompt_sen_1 = data["input_ids_prompt_sen_1"].to(device)
        attention_masks_prompt_sen_1 = data["attention_masks_prompt_sen_1"].to(device)
        input_ids_template_sen_1 = data["input_ids_template_sen_1"].to(device)
        attention_masks_template_sen_1 = data["attention_masks_template_sen_1"].to(device)

        input_ids_prompt_sen_2 = data["input_ids_prompt_sen_2"].to(device)
        attention_masks_prompt_sen_2 = data["attention_masks_prompt_sen_2"].to(device)
        input_ids_template_sen_2 = data["input_ids_template_sen_2"].to(device)
        attention_masks_template_sen_2 = data["attention_masks_template_sen_2"].to(device)

        labels = data["label"].to(device)
        with torch.no_grad():
            # 前向传播
            sentence_embedding_a = lora.forward(input_ids_prompt_sen_1, attention_masks_prompt_sen_1
                                                            , input_ids_template_sen_1, attention_masks_template_sen_1)
            sentence_embedding_b = lora.forward(input_ids_prompt_sen_2, attention_masks_prompt_sen_2
                                                            , input_ids_template_sen_2, attention_masks_template_sen_2)

        similarity = F.cosine_similarity(sentence_embedding_a, sentence_embedding_b, dim=-1)
        similarity = similarity.detach().cpu().numpy()
        pre.append(similarity)
        label.append(labels.cpu().numpy())

    pre = torch.tensor(pre)
    pre = (pre >= thres_hold).long().detach().cpu().numpy()
    acc = accuracy_score(pre, label)
    print('dev acc:', acc)
    print()


if __name__ == '__main__':

    seed_everything()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = init_tokenizer("roberta_data")

    lora = Lora_Model('roberta_data', tokenizer).to(device)
    epoch = 10
    lr = 3e-5

    optimzer = torch.optim.AdamW(lora.parameters(), lr=lr)
    train_data_loader, dev_data_loader = load_data(tokenizer, 2)
    train()
    infer()
