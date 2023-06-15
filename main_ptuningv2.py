from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import  DataLoader,Dataset
from torch.optim import AdamW
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from prefix_encoder import PrefixEncoder
import pandas as pd


def seed_everything(seed=3427):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


auto_config = AutoConfig.from_pretrained("roberta_data")


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sentence_embedding = torch.sum(input_mask * last_hidden_state, dim=1)
        sum_mask = torch.sum(input_mask, dim=1)
        sum_mask = torch.clip(sum_mask, min=1e-8)
        mean_sentence_embedding = sentence_embedding / sum_mask
        return mean_sentence_embedding


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


class P_Tuningv2_Model(nn.Module):
    def __init__(self, ptv2_cfg):
        super().__init__()
        self.ptv2_cfg = ptv2_cfg
        self.prefix_encoder = PrefixEncoder(self.ptv2_cfg)
        self.bert = AutoModel.from_pretrained("roberta_data")
        self.dropout = torch.nn.Dropout(self.ptv2_cfg.hidden_dropout_prob)
        self.meanpooling = MeanPooling()
        # self.tokenizer = tokenizer
        # self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        # #### 切记已经加入这个【x】这个token  2w _>20001
        # self.bert.resize_token_embeddings(len(self.tokenizer))

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = self.ptv2_cfg.pre_seq_len
        self.n_layer = self.ptv2_cfg.num_hidden_layers
        self.n_head = self.ptv2_cfg.num_attention_heads
        self.n_embd = self.ptv2_cfg.hidden_size // self.ptv2_cfg.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder.forward(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        # (128, 12 * 2 * 768) -> (batch,128,24,12,)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        # [2, 0, 3, 1, 4] means [n_layer*2,batch_size,n_head,pre_seq_len,n_embd]
        # split(2) nlayer 个 [n_layer,batch_size,n_head,pre_seq_len,n_embd]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,input_ids,attention_mask,mode='train'):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        bert_output1, bert_output2 = self.bert(input_ids, attention_mask, past_key_values=past_key_values,
                                               return_dict=False)
        sentence_embedding = self.meanpooling(bert_output1,attention_mask[:,self.pre_seq_len:])
        if mode == 'train':
            loss = self.cal_loss(sentence_embedding)
            return loss
        else:
            return sentence_embedding


    def cal_loss(self, sentence_embedding, tao=0.05, device='cuda'):
        # [2*b,hidden_size]
        # idxs [0-5]
        idxs = torch.arange(0, sentence_embedding.shape[0], device=device)
        y_ture = idxs + 1 - idxs % 2 * 2
        # 构建余弦相似度的矩阵
        sim = F.cosine_similarity(sentence_embedding.unsqueeze(1), sentence_embedding.unsqueeze(0), dim=-1)
        # 将对角线上的设置为负无穷
        sim = sim - torch.eye(sentence_embedding.shape[0], device=device) * 1e10

        # 除以温度系数
        sim = sim / tao
        loss = F.cross_entropy(sim, y_ture)
        return torch.mean(loss)

class MyDtasets(Dataset):
    # 将train数据转化为list，collate_fn负责复制句子
    def __init__(self,df,tokenizer,max_len=128,mode='train'):
        super(MyDtasets, self).__init__()
        self.df = df
        self.max_len = max_len
        self.mode = mode
        self.tokenizer = tokenizer
        if self.mode == 'train':
            self.datalist = self.df2list(self.df)



    def df2list(self,df):
        #将train的df转化为list
        text_a = df['text_a'].tolist()
        text_b = df['text_b'].tolist()
        text_a.extend(text_b)
        return text_a
    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.datalist[index]

            inputs = self.tokenizer(data,truncation = True,max_length=self.max_len)
            return {
                'input_id':torch.as_tensor(inputs['input_ids'],dtype=torch.long),
                'attention_mask':torch.as_tensor(inputs['attention_mask'],dtype=torch.long)
            }
        else:
            data = self.df.loc[index]
            text_a = data['text_a']
            text_b = data['text_b']
            label = data['label']

            inputs_a = self.tokenizer(text_a, truncation=True, max_length=self.max_len)
            inputs_b = self.tokenizer(text_b, truncation=True, max_length=self.max_len)


            return{
                'input_id_a':torch.as_tensor(inputs_a['input_ids'],dtype=torch.long),
                 'attention_mask_a':torch.as_tensor(inputs_a['attention_mask'],dtype=torch.long),
                'input_id_b':torch.as_tensor(inputs_b['input_ids'],dtype=torch.long),
                 'attention_mask_b':torch.as_tensor(inputs_b['attention_mask'],dtype=torch.long),
                'label':torch.as_tensor(label,dtype=torch.long)
            }


    def __len__(self):
        if self.mode == 'train':
            return  len(self.datalist)
        else:
            return len(self.df)


def collate_fn_dev(batch):
    # [11,12,13]
    max_len_a = max([len(x['input_id_a'])for x in batch])
    max_len_b = max([len(x['input_id_b'])for x in batch])

    # [1,2,3] seq
    #
    # [3,3]  [111,0,0]  [111,123,0]

    input_ids_a = torch.zeros(len(batch),max_len_a,dtype=torch.long)
    input_ids_b = torch.zeros(len(batch),max_len_b,dtype=torch.long)

    attention_masks_a = torch.zeros(len(batch),max_len_a,dtype=torch.long)
    attention_masks_b = torch.zeros(len(batch),max_len_b,dtype=torch.long)
    labels = []
    for i,x in enumerate(batch):
        input_ids_a[i,:len(x['input_id_a'])] = x['input_id_a']
        attention_masks_a[i,:len(x['attention_mask_a'])] = x['attention_mask_a']
        input_ids_b[i, :len(x['input_id_b'])] = x['input_id_b']
        attention_masks_b[i, :len(x['attention_mask_b'])] = x['attention_mask_b']
        labels.append(x['label'])

    return{
        'input_ids_a':input_ids_a,
        'attention_masks_a':attention_masks_a,
        'input_ids_b':input_ids_b,
        'attention_masks_b':attention_masks_b,
        'labels':torch.tensor(labels,dtype=torch.long)


    }

def collate_fn_train(batch):
    max_len = max([len(x['input_id']) for x  in  batch])
    # [A,B,C] ->[A,A,B,B,C,C]   0 ~0-1    1 ~2-3     2  4-5    idx ->  2*idx  2*idx+1
    input_ids = torch.zeros(2*len(batch), max_len,dtype=torch.long)
    attention_masks = torch.zeros(2*len(batch), max_len,dtype=torch.long)
    for i,x in enumerate(batch):
        input_ids[2*i,:len(x['input_id'])]  = x['input_id']
        attention_masks[2*i,:len(x['attention_mask'])]  = x['attention_mask']
        input_ids[2*i+1,:len(x['input_id'])]  = x['input_id']
        attention_masks[2*i+1,:len(x['attention_mask'])]  = x['attention_mask']

    return {
        'input_ids':input_ids,
        'attention_masks':attention_masks

    }

def load_data(batch_size=32):
    train_df =  pd.read_csv(os.path.join("data","ants","train.csv"))
    dev_df = pd.read_csv(os.path.join("data","ants","dev.csv"))

    tokenizer = AutoTokenizer.from_pretrained("roberta_data")

    train_sets = MyDtasets(train_df,tokenizer,mode='train')
    train_loader = DataLoader(train_sets,batch_size,collate_fn=collate_fn_train,shuffle=False)

    dev_sets = MyDtasets(dev_df, tokenizer, mode='dev')
    dev_loader = DataLoader(dev_sets, batch_size, collate_fn=collate_fn_dev, shuffle=True)



    return  train_loader,dev_loader

def train(epochs = 5,lr=3e-5,threshold=0.5):

    seed_everything()

    optimizer = AdamW(model.parameters(),lr)

    best_acc = 0

    for epoch in range(epochs):
        print('epoch',epoch+1)

        model.train()

        pbar = tqdm(train_loader)

        for data in pbar:

            optimizer.zero_grad()
            input_ids = data['input_ids'].to(device)
            attenion_masks =  data['attention_masks'].to(device)
            loss = model(input_ids,attenion_masks,mode='train')
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_description(f'loss:{loss.item():.4f}')

        pre = []
        label = []
        model.eval()

        for data in tqdm(dev_loader):
            input_ids_a = data['input_ids_a'].to(device)
            attention_masks_a = data['attention_masks_a'].to(device)
            input_ids_b = data['input_ids_b'].to(device)
            attention_masks_b = data['attention_masks_b'].to(device)

            labels = data['labels'].to(device)

            with torch.no_grad():
                setence_emb_a= model(input_ids_a,attention_masks_a,mode='dev')
                setence_emb_b= model(input_ids_b,attention_masks_b,mode='dev')

            sim = F.cosine_similarity(setence_emb_a,setence_emb_b,dim=-1)

            sim = sim.detach().cpu().numpy()
            pre.extend(sim)

            label.extend(labels.detach().cpu().numpy())

        pre = torch.tensor(pre)
        # [0.9,0.2]->tensor ->[true,false]->long->[1,0]
        pre = (pre>=threshold).long().detach().cpu().numpy()

        acc = accuracy_score(label,pre)

        print('dev_acc:',acc)
        print()

        if acc>best_acc:
            torch.save(model.state_dict(),'./model_weight/simcse.bin')
            best_acc = acc
def infer(threshold=0.5):
    print('ptuningv2-simcse开始推理')
    pre = []
    label = []
    model.load_state_dict(torch.load('./model_weight/simcse.bin',map_location=device))
    model.eval()

    for data in tqdm(dev_loader):
        input_ids_a = data['input_ids_a'].to(device)
        attenion_masks_a = data['attenion_masks_a'].to(device)
        input_ids_b = data['input_ids_b'].to(device)
        attenion_masks_b = data['attenion_masks_b'].to(device)

        labels = data['labels'].to(device)

        with torch.no_grad():
            setence_emb_a = model(input_ids_a, attenion_masks_a, mode='dev')
            setence_emb_b = model(input_ids_b, attenion_masks_b, mode='dev')

        sim = F.cosine_similarity(setence_emb_a, setence_emb_b, dim=-1)

        sim = sim.detach().cpu().numpy()
        pre.extend(sim)

        label.extend(labels.detach().cpu().numpy())

    pre = torch.tensor(pre)
    # [0.9,0.2]->tensor ->[true,false]->long->[1,0]
    pre = (pre >= threshold).long().detach().cpu().numpu()

    acc = accuracy_score(label, pre)

    print(' infer dev_acc:', acc)
    print()

if __name__ == '__main__':

    pretrained_path = "roberta_data"
    ptv2_cfg = p_tuningv2_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch = 10
    model = P_Tuningv2_Model(ptv2_cfg).to(device)
    train_loader, dev_loader = load_data(28)
    train(epochs=10,lr=3e-5,threshold=0.6)
    print("inferring")
    infer(threshold=0.6)