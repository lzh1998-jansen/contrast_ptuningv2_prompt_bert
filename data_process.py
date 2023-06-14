import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os


class MyDatasets(Dataset):
    def __init__(self, df, tokenizer, max_len=64, mode='train'):
        super(MyDatasets, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.datalist = self.df2list(df)

    def df2list(self, df):
        text_a = df['text_a'].values.tolist()
        text_b = df['text_b'].values.tolist()
        return text_a + text_b

    def __getitem__(self, index):
        if self.mode == 'train':
            # 我爱你
            data = self.datalist[index]
            # 我爱你这句话的意思是[MASK]  h   [x][x][x]这句话的意思是[MASK]  h^   h' 我爱你，他的意思是[MASK] h'^ [x][x][x]，他的意思是[MASK]
            sentences_tem = self.template(data)

            prompt_sen_1 = sentences_tem[0]
            template_sen_1 = sentences_tem[1]

            prompt_sen_2 = sentences_tem[2]
            template_sen_2 = sentences_tem[3]

            inputs_prompt_sen_1 = self.tokenizer(prompt_sen_1, truncation=True, max_length=self.max_len)
            inputs_template_sen_1 = self.tokenizer(template_sen_1, truncation=True, max_length=self.max_len)

            inputs_prompt_sen_2 = self.tokenizer(prompt_sen_2, truncation=True, max_length=self.max_len)
            inputs_template_sen_2 = self.tokenizer(template_sen_2, truncation=True, max_length=self.max_len)

            return {
                'input_id_prompt_sen_1': torch.as_tensor(inputs_prompt_sen_1['input_ids'], dtype=torch.long),
                'attention_mask_prompt_sen_1': torch.as_tensor(inputs_prompt_sen_1['attention_mask'], dtype=torch.long),

                'input_id_template_sen_1': torch.as_tensor(inputs_template_sen_1['input_ids'], dtype=torch.long),
                'attention_mask_template_sen_1': torch.as_tensor(inputs_template_sen_1['attention_mask'],
                                                                 dtype=torch.long),
                'input_id_prompt_sen_2': torch.as_tensor(inputs_prompt_sen_2['input_ids'], dtype=torch.long),
                'attention_mask_prompt_sen_2': torch.as_tensor(inputs_prompt_sen_2['attention_mask'], dtype=torch.long),

                'input_id_template_sen_2': torch.as_tensor(inputs_template_sen_2['input_ids'], dtype=torch.long),
                'attention_mask_template_sen_2': torch.as_tensor(inputs_template_sen_2['attention_mask'],
                                                                 dtype=torch.long)
            }


        else:
            """
            去实现一下将一句话过模板  【X】这句话代表的意思是 [MASK]  h  h^     h-h^   h
            """
            data = self.df.iloc[index]
            text_a = data['text_a']
            text_b = data['text_b']
            label = data['label']
            # text_a 生成的模板
            sentences_tem_a = self.single_template(text_a)
            # text_b 生成的模板
            sentences_tem_b = self.single_template(text_b)

            # 带入模板之后的 a,b
            prompt_sen_1_a = sentences_tem_a[0]
            template_sen_1_a = sentences_tem_a[1]
            prompt_sen_1_b = sentences_tem_b[0]
            template_sen_1_b = sentences_tem_b[1]

            # 分词
            input_prompt_sen_1_a = self.tokenizer(prompt_sen_1_a, truncation=True, max_length=self.max_len)
            input_template_sen_1_a = self.tokenizer(template_sen_1_a, truncation=True, max_length=self.max_len)
            input_prompt_sen_1_b = self.tokenizer(prompt_sen_1_b, truncation=True, max_length=self.max_len)
            input_template_sen_1_b = self.tokenizer(template_sen_1_b, truncation=True, max_length=self.max_len)

            return {
                "input_ids_prompt_sen_1_a": torch.as_tensor(input_prompt_sen_1_a["input_ids"], dtype=torch.long),
                "attention_mask_prompt_sen_1_a": torch.as_tensor(input_prompt_sen_1_a["attention_mask"],
                                                                 dtype=torch.long),

                "input_ids_template_sen_1_a": torch.as_tensor(input_template_sen_1_a["input_ids"], dtype=torch.long),
                "attention_mask_template_sen_1_a": torch.as_tensor(input_template_sen_1_a["attention_mask"],
                                                                   dtype=torch.long),

                "input_ids_prompt_sen_1_b": torch.as_tensor(input_prompt_sen_1_b["input_ids"], dtype=torch.long),
                "attention_mask_prompt_sen_1_b": torch.as_tensor(input_prompt_sen_1_b["attention_mask"],
                                                                 dtype=torch.long),

                "input_ids_template_sen_1_b": torch.as_tensor(input_template_sen_1_b["input_ids"], dtype=torch.long),
                "attention_mask_template_sen_1_b": torch.as_tensor(input_template_sen_1_b["attention_mask"],
                                                                   dtype=torch.long),

                "label": torch.as_tensor(label, dtype=torch.long)
            }

    def single_template(self, sentence):
        prompt_tem = ['[X]，这句话的意思是[MASK]']
        sentence_tem = []
        for template in prompt_tem:
            prompt_sentence = template.replace("[X]", sentence)
            words_len = len(sentence)
            template_senence = template.replace("[X]", "[X]" * words_len)
            sentence_tem += [prompt_sentence, template_senence]

        return sentence_tem

    def template(self, sentence):
        prompt_tem = ['[X]，这句话的意思是[MASK]', '[X],它的意思是[MASK]']
        sentence_tem = []
        for template in prompt_tem:
            prompt_setence = template.replace('[X]', sentence)
            words_len = len(sentence)
            template_sentence = template.replace('[X]', '[X]' * words_len)
            sentence_tem += [prompt_setence, template_sentence]

        return sentence_tem

    def __len__(self):
        if self.mode == 'train':
            return len(self.datalist)
        else:
            return len(self.df)


def collate_fn_train(batch):
    max_prompt_len_sen_1 = max([len(x['input_id_prompt_sen_1']) for x in batch])
    #
    max_template_len_sen_1 = max([len(x['input_id_template_sen_1']) for x in batch])

    max_prompt_len_sen_2 = max([len(x['input_id_prompt_sen_2']) for x in batch])
    max_template_len_sen_2 = max([len(x['input_id_template_sen_2']) for x in batch])

    input_ids_prompt_sen_1 = torch.zeros(len(batch), max_prompt_len_sen_1, dtype=torch.long)
    attention_masks_prompt_sen_1 = torch.zeros(len(batch), max_prompt_len_sen_1, dtype=torch.long)

    input_ids_template_sen_1 = torch.zeros(len(batch), max_template_len_sen_1, dtype=torch.long)
    attention_masks_template_sen_1 = torch.zeros(len(batch), max_template_len_sen_1, dtype=torch.long)

    input_ids_prompt_sen_2 = torch.zeros(len(batch), max_prompt_len_sen_2, dtype=torch.long)
    attention_masks_prompt_sen_2 = torch.zeros(len(batch), max_prompt_len_sen_2, dtype=torch.long)

    input_ids_template_sen_2 = torch.zeros(len(batch), max_template_len_sen_2, dtype=torch.long)
    attention_masks_template_sen_2 = torch.zeros(len(batch), max_template_len_sen_2, dtype=torch.long)

    for i, x in enumerate(batch):
        input_ids_prompt_sen_1[i, :len(x['input_id_prompt_sen_1'])] = x['input_id_prompt_sen_1']
        attention_masks_prompt_sen_1[i, :len(x['attention_mask_prompt_sen_1'])] = x['attention_mask_prompt_sen_1']

        input_ids_template_sen_1[i, :len(x['input_id_template_sen_1'])] = x['input_id_template_sen_1']
        attention_masks_template_sen_1[i, :len(x['attention_mask_template_sen_1'])] = x['attention_mask_template_sen_1']

        input_ids_prompt_sen_2[i, :len(x['input_id_prompt_sen_2'])] = x['input_id_prompt_sen_2']
        attention_masks_prompt_sen_2[i, :len(x['attention_mask_prompt_sen_2'])] = x['attention_mask_prompt_sen_2']

        input_ids_template_sen_2[i, :len(x['input_id_template_sen_2'])] = x['input_id_template_sen_2']
        attention_masks_template_sen_2[i, :len(x['attention_mask_template_sen_2'])] = x['attention_mask_template_sen_2']

    return {
        'input_ids_prompt_sen_1': input_ids_prompt_sen_1,
        'attention_masks_prompt_sen_1': attention_masks_prompt_sen_1,

        'input_ids_template_sen_1': input_ids_template_sen_1,
        'attention_masks_template_sen_1': attention_masks_template_sen_1,

        'input_ids_prompt_sen_2': input_ids_prompt_sen_2,
        'attention_masks_prompt_sen_2': attention_masks_prompt_sen_2,

        'input_ids_template_sen_2': input_ids_template_sen_2,
        'attention_masks_template_sen_2': attention_masks_template_sen_2,

    }


def collate_fn_dev(batch):
    max_prompt_len_sen_1 = max([len(x['input_ids_prompt_sen_1_a']) for x in batch])
    #
    max_template_len_sen_1 = max([len(x['input_ids_template_sen_1_a']) for x in batch])

    max_prompt_len_sen_2 = max([len(x['input_ids_prompt_sen_1_b']) for x in batch])
    max_template_len_sen_2 = max([len(x['input_ids_template_sen_1_b']) for x in batch])

    input_ids_prompt_sen_1 = torch.zeros(len(batch), max_prompt_len_sen_1, dtype=torch.long)
    attention_masks_prompt_sen_1 = torch.zeros(len(batch), max_prompt_len_sen_1, dtype=torch.long)

    input_ids_template_sen_1 = torch.zeros(len(batch), max_template_len_sen_1, dtype=torch.long)
    attention_masks_template_sen_1 = torch.zeros(len(batch), max_template_len_sen_1, dtype=torch.long)

    input_ids_prompt_sen_2 = torch.zeros(len(batch), max_prompt_len_sen_2, dtype=torch.long)
    attention_masks_prompt_sen_2 = torch.zeros(len(batch), max_prompt_len_sen_2, dtype=torch.long)

    input_ids_template_sen_2 = torch.zeros(len(batch), max_template_len_sen_2, dtype=torch.long)
    attention_masks_template_sen_2 = torch.zeros(len(batch), max_template_len_sen_2, dtype=torch.long)

    label = []
    for i, x in enumerate(batch):
        input_ids_prompt_sen_1[i, :len(x['input_ids_prompt_sen_1_a'])] = x['input_ids_prompt_sen_1_a']
        attention_masks_prompt_sen_1[i, :len(x['attention_mask_prompt_sen_1_a'])] = x['attention_mask_prompt_sen_1_a']

        input_ids_template_sen_1[i, :len(x['input_ids_template_sen_1_a'])] = x['input_ids_template_sen_1_a']
        attention_masks_template_sen_1[i, :len(x['attention_mask_template_sen_1_a'])] = x[
            'attention_mask_template_sen_1_a']

        input_ids_prompt_sen_2[i, :len(x['input_ids_prompt_sen_1_b'])] = x['input_ids_prompt_sen_1_b']
        attention_masks_prompt_sen_2[i, :len(x['attention_mask_prompt_sen_1_b'])] = x['attention_mask_prompt_sen_1_b']

        input_ids_template_sen_2[i, :len(x['input_ids_template_sen_1_b'])] = x['input_ids_template_sen_1_b']
        attention_masks_template_sen_2[i, :len(x['attention_mask_template_sen_1_b'])] = x[
            'attention_mask_template_sen_1_b']
        label.append(x['label'])
    return {
        'input_ids_prompt_sen_1': input_ids_prompt_sen_1,
        'attention_masks_prompt_sen_1': attention_masks_prompt_sen_1,

        'input_ids_template_sen_1': input_ids_template_sen_1,
        'attention_masks_template_sen_1': attention_masks_template_sen_1,

        'input_ids_prompt_sen_2': input_ids_prompt_sen_2,
        'attention_masks_prompt_sen_2': attention_masks_prompt_sen_2,

        'input_ids_template_sen_2': input_ids_template_sen_2,
        'attention_masks_template_sen_2': attention_masks_template_sen_2,
        'label': torch.tensor(label, dtype=torch.long)

    }


def load_data(tokenizer, batch_size=32):
    train_df = pd.read_csv('data/ants/train.csv')
    train_sets = MyDatasets(train_df, tokenizer, 64, mode='train')

    train_dataloader = DataLoader(train_sets, batch_size, collate_fn=collate_fn_train, shuffle=True)

    dev_df = pd.read_csv('data/ants/dev.csv')
    dev_sets = MyDatasets(dev_df, tokenizer, 64, mode='dev')

    dev_dataloader = DataLoader(dev_sets, batch_size, collate_fn=collate_fn_dev, shuffle=False)

    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('roberta_data')
    tokenizer.add_tokens(['[X]'])
    train_dataloader, dev_dataloader = load_data(tokenizer)
    for batch in dev_dataloader:
        print(batch['input_ids_prompt_sen_1'].shape)
        break