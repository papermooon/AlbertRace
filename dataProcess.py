import torch
from transformers import AlbertTokenizer, AlbertForMultipleChoice
from datasets import *

# data_path = './racedata'
# data.save_to_disk(data_path)
# print(datasets.list_datasets())

data = load_dataset('race', 'all')
print(data['test'].features)
print(ord(data['test'][1]['answer'])-65)

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForMultipleChoice.from_pretrained('albert-base-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 定义数据集
class RaceDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 答案字母转成数字
        label = ord(self.data[idx]['answer'])-65
        question = self.data[idx]['question']
        content = '。'.join(self.data[idx][0])
        choice = self.data[idx][1][0]['choice']
        if len(choice) < 4:  # 如果选项不满四个，就补“不知道”
            for i in range(4 - len(choice)):
                choice.append('不知道')

        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i for i in choice]

        return content, pair, label


def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True,
                         max_length=128, return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label
