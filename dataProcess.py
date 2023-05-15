import torch
from transformers import AlbertTokenizer, AlbertForMultipleChoice
from datasets import *
from torch.utils.data import Dataset, DataLoader, TensorDataset

# data_path = './racedata'
# data.save_to_disk(data_path)

data = load_dataset('race', 'all')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')


# 定义数据集
class RaceDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 答案字母转成数字
        label = ord(self.data[idx]['answer']) - 65
        question = self.data[idx]['question']
        content = self.data[idx]['article']
        choice = self.data[idx]['options']
        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i for i in choice]

        return content, pair, label


def collate_fn(batch_data):  # 问题对在前，文章内容在后，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids, label = [], [], [], []
    for x in batch_data:
        text = tokenizer(x[0], x[1], padding='max_length', truncation=True,
                         max_length=512, return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
        label.append(x[-1])
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    # print(x[-1])
    # label = torch.tensor([x[-1] for x in data])
    label = torch.tensor(label)
    return input_ids, attention_mask, token_type_ids, label

# res = collate_fn(tds)
# print(res[3])
# print(res[0].shape)
# print(res[1].shape)
# print(res[2].shape)
