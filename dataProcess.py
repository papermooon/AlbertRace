import torch
from transformers import AlbertTokenizer, AlbertForMultipleChoice
from datasets import *
from torch.utils.data import Dataset, DataLoader, TensorDataset

data_path = './racedata'
# data.save_to_disk(data_path)

data = load_dataset('race', 'all')
# data = load_dataset(path=data_path)
tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')


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
        pair = [question + i for i in choice]

        return content, pair, label


def collate_fn(batch_data):  # 问题对在前，文章内容在后，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids, label = [], [], [], []
    for x in batch_data:
        text = tokenizer(x[0], x[1], padding='max_length', truncation='longest_first',
                         max_length=256, return_tensors='pt')
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

# tds = RaceDataset(data['train'])[2]
# tds2 = RaceDataset(data['train'])[3]
# print(tds)
# res = collate_fn([tds])
# print(res[3])
# print(res[0].shape)
# print(res[1].shape)
# print(res[2].shape)
# decode1=tokenizer.decode(res[0][0][0])
# decode2=tokenizer.decode(res[0][0][1])
# decode3=tokenizer.decode(res[0][0][2])
# decode4=tokenizer.decode(res[0][0][3])
# print(decode1)
# print(decode2)
# print(decode3)
# print(decode4)
# labels = torch.tensor(0).unsqueeze(0)
# print(labels)
