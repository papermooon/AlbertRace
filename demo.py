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


def sample(Content, Question, Options, Answer, model):
    input_ids = []

    label = ord(Answer) - 65
    content = [Content for i in range(len(Options))]
    pair = [Question + i for i in Options]
    encoding = dataProcess.tokenizer(content, pair, padding='max_length', truncation='longest_first',
                                     max_length=150, return_tensors='pt')

    input_ids.append(encoding['input_ids'].tolist())
    input_ids = torch.tensor(input_ids).to(device)

    outputs = model(input_ids)

    result = torch.argmax(outputs.logits).detach().cpu().numpy()
    # print("模型答案：", result)
    # print("标准答案：", label)
    return result == label


test_dataset = dataProcess.data['test']
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=dataProcess.collate_fn)


def model_eval(checkpoint_name):
    correct = 0
    model = torch.load(model_dir / checkpoint_name)
    print("加载模型:", checkpoint_name)
    model.to(device)
    model.eval()
    tk = tqdm(enumerate(test_dataset), total=len(test_dataset), position=0, leave=True)
    for idx, samData in tk:
        res = sample(samData['article'], samData['question'], samData['options'], samData['answer'], model)
        if res:
            correct += 1
    return correct / len(test_dataset)


print(model_eval('epoch_4_05-18_05-27.pt'))
