from transformers import AutoConfig,AutoTokenizer
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from model import P_Tuningv2_Model,compute_loss
from data_process import  load_data
from sklearn.metrics import accuracy_score

def seed_everything(seed=3427):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def init_tokenizer(path):
    # [X]
    tokenizer = AutoTokenizer.from_pretrained(path)

    tokenizer.add_tokens(['[X]'])
    return tokenizer

auto_config = AutoConfig.from_pretrained("roberta_data")


class p_tuningv2_config():
    # ptuning parameter
    prefix_projection = True
    pre_seq_len = 128
    prefix_hidden_size = 768
    # roberta parameter
    hidden_size = auto_config.hidden_size
    num_hidden_layers = auto_config.num_hidden_layers
    num_attention_heads = auto_config.num_attention_heads
    hidden_dropout_prob = auto_config.hidden_dropout_prob


def train():
    best_acc = 0
    for e in range(epoch):
        print(f"当前是第{e}轮")
        model.train()
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
            s1 = model.forward(input_ids_prompt_sen_1, attention_masks_prompt_sen_1, input_ids_template_sen_1,
                               attention_masks_template_sen_1)
            # h_i'-h_i'^
            s2 = model.forward(input_ids_prompt_sen_2, attention_masks_prompt_sen_2, input_ids_template_sen_2,
                               attention_masks_template_sen_2)
            loss = compute_loss(s1,s2)
            loss.backward()
            optimzer.step()
            pbar.update()
            pbar.set_description(f"当前loss={loss.item():.6f}")

        pred = []
        label = []
        model.eval()
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
                sentence_embedding_a = model.forward(input_ids_prompt_sen_1, attention_masks_prompt_sen_1,
                                                     input_ids_template_sen_1, attention_masks_template_sen_1)
                sentence_embedding_b = model.forward(input_ids_prompt_sen_2, attention_masks_prompt_sen_2,
                                                     input_ids_template_sen_2, attention_masks_template_sen_2)

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
            torch.save(model.state_dict(), './model_weight/pt2_prompt.bin')
            best_acc = acc

def infer(thres_hold=0.5):
    # t1->emb->cos
    print(f"prompt bert 开始推理")

    model.load_state_dict(torch.load('./model_weight/pt2_prompt.bin' ,map_location=device))
    model.eval()

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
            sentence_embedding_a = model.forward(input_ids_prompt_sen_1, attention_masks_prompt_sen_1
                                                 ,input_ids_template_sen_1 ,attention_masks_template_sen_1)
            sentence_embedding_b = model.forward(input_ids_prompt_sen_2, attention_masks_prompt_sen_2
                                                 ,input_ids_template_sen_2 ,attention_masks_template_sen_2)

        similarity = F.cosine_similarity(sentence_embedding_a ,sentence_embedding_b ,dim=-1)
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
    pretrained_path = "roberta_data"
    tokenizer = init_tokenizer(pretrained_path)
    ptv2_cfg = p_tuningv2_config()
    train_data_loader, dev_data_loader = load_data(tokenizer, 2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = P_Tuningv2_Model(ptv2_cfg,tokenizer).to(device)
    print(model)

    # train args
    epoch = 10
    lr = 3e-5
    optimzer = torch.optim.AdamW(model.parameters(), lr=lr)

    train()
    infer()
    pass
