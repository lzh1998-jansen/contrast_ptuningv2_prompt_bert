import torch
from torch import nn
import torch.nn.functional as F
from prefix_encoder import  PrefixEncoder
from transformers import AutoModel,AutoTokenizer
from copy import deepcopy
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,last_hidden_state,attention_mask):
        input_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sentence_embedding = torch.sum(input_mask*last_hidden_state,dim=1)
        sum_mask = torch.sum(input_mask,dim=1)
        sum_mask = torch.clip(sum_mask,min=1e-8)
        mean_sentence_embedding = sentence_embedding/sum_mask
        return  mean_sentence_embedding




class P_Tuningv2_Model(nn.Module):
    def __init__(self,ptv2_cfg,tokenizer):
        super().__init__()
        self.ptv2_cfg = ptv2_cfg
        self.prefix_encoder = PrefixEncoder(self.ptv2_cfg)
        self.bert = AutoModel.from_pretrained("roberta_data")
        self.dropout = torch.nn.Dropout(self.ptv2_cfg.hidden_dropout_prob)
        self.meanpooling = MeanPooling()
        self.tokenizer = tokenizer
        self.mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        #### 切记已经加入这个【x】这个token  2w _>20001
        self.bert.resize_token_embeddings(len(self.tokenizer))

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

    def get_prompt(self,batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size,-1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        # [2, 0, 3, 1, 4] means [n_layer*2,batch_size,n_head,pre_seq_len,n_embd]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    """
        def forward(self,input_ids,attention_mask):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        bert_output1,bert_output2 = self.bert(input_ids,attention_mask,past_key_values=past_key_values,return_dict=False)
        sentence_embedding = self.dropout(bert_output1)
        mean_embedding = self.meanpooling(sentence_embedding,attention_mask[:,self.pre_seq_len:])
        return mean_embedding
    """
    def forward(self, prompt_input_ids, prompt_attention_mask,  template_input_ids,template_attention_mask):
        prompt_mask_emb = self.cal_mask_embedding(prompt_input_ids,prompt_attention_mask)
        template_mask_emb = self.cal_mask_embedding(template_input_ids,template_attention_mask)
        return  prompt_mask_emb-template_mask_emb

    def cal_mask_embedding(self,input_ids,attention_mask):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        bert_output1, bert_output2 = self.bert(input_ids, attention_mask, past_key_values=past_key_values,
                                               return_dict=False)
        bert_output1 = self.dropout(bert_output1)
        mask_index = (input_ids == self.mask_id).long()
        #  do not forget the data type should changed to be float .
        input_mask_expand = mask_index.unsqueeze(-1).expand(bert_output1.size()).float()
        mask_embedding = torch.sum(bert_output1*input_mask_expand,dim=1)
        return mask_embedding


def compute_loss(query, key, tao=0.05):
    # query = h_i - h_i^     (n,d)
    # key  = h_i' - h_i'^

    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)

    N, D = query.shape[0], query.shape[1]

    # 分子部分  cos(query,key)  [n,1]
    # (N,1,1)->(N,1)
    batch_pos = torch.exp(torch.div(torch.bmm(query.view(N, 1, D), key.view(N, D, 1)).view(N, 1), tao))

    # 分母部分 计算查询向量和所有键向量的相似度  q1 -> key   (n,d)  (d,n) (n,n)
    batch_all = torch.sum(torch.exp(torch.div(torch.mm(query, torch.t(key)), tao)), dim=1)

    loss = torch.mean(-torch.log(torch.div(batch_pos, batch_all)))
    return loss
