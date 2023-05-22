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
    token_type_ids = []

    label = ord(Answer) - 65
    content = [Content for i in range(len(Options))]
    pair = [Question + i for i in Options]
    encoding = dataProcess.tokenizer(content, pair, padding='max_length', truncation='longest_first',
                                     max_length=256, return_tensors='pt')

    input_ids.append(encoding['input_ids'].tolist())
    input_ids = torch.tensor(input_ids).to(device)

    token_type_ids.append(encoding['token_type_ids'].tolist())
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.to(device)

    outputs = model(input_ids, token_type_ids=token_type_ids)

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
        tk.set_postfix(acc=correct / (idx + 1), idx=idx)
    return correct / len(test_dataset)


# print(model_eval('epoch_0_05-17_20-33.pt'))
# print(model_eval('epoch_0_05-19_01-23.pt'))   57.2
# print(model_eval('epoch_1_05-19_03-31.pt'))   58.9
# print(model_eval('epoch_2_05-19_05-40.pt'))   59.2
# print(model_eval('epoch_3_05-19_07-49.pt'))   58.9
# print(model_eval('epoch_4_05-19_09-57.pt'))   59.0

# print(model_eval('epoch_0_05-19_01-23.pt')) 59.0
# print(model_eval('epoch_1_05-19_03-31.pt')) 61.1
# print(model_eval('epoch_2_05-19_05-40.pt')) 60.6
# print(model_eval('epoch_3_05-19_07-49.pt')) 62.0
# print(model_eval('epoch_4_05-19_09-57.pt')) 62.5
