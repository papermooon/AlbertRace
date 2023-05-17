import torch
from tqdm import tqdm
from transformers import AlbertForMultipleChoice
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
import dataProcess
from torch.utils.tensorboard import SummaryWriter
from transformers import logging

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = Path("./model")
# checkpoint = "epoch_2_05-15_23-30.pt"
checkpoint = "epoch_0_05-17_02-42.pt"

model = torch.load(model_dir / checkpoint)
print("加载模型:", checkpoint)
model.to(device)
model.eval()


def sample(Content, Question, Options, Answer):
    input_ids = []
    # attention_mask = []
    label = ord(Answer) - 65
    content = [Content for i in range(len(Options))]
    pair = [Question + i for i in Options]
    encoding = dataProcess.tokenizer(content, pair, padding='max_length', truncation='longest_first',
                                     max_length=150, return_tensors='pt')

    input_ids.append(encoding['input_ids'].tolist())
    input_ids = torch.tensor(input_ids).to(device)
    print(input_ids.shape)
    # attention_mask.append(encoding['attention_mask'].tolist())
    # attention_mask = torch.tensor(attention_mask).to(device)
    outputs = model(input_ids)
    print(outputs)
    result = torch.argmax(outputs.logits).detach().cpu().numpy()
    print("模型答案：", result)
    print("标准答案：", label)
    return result == label


train_dataset = dataProcess.data['test']
ct = 0
for i in range(500):
    samData = train_dataset[i]
    # print(samData)
    res = sample(samData['article'], samData['question'], samData['options'], samData['answer'])
    if res:
        ct += 1
print(ct / 500)
