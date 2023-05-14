import torch
from tqdm import tqdm
from transformers import AlbertForMultipleChoice
from pathlib import Path
from torch.utils.data import DataLoader
import dataProcess

# 超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = Path("./model")
checkpoint = None
batch_size = 128
epochs = 20


# loss的进度条
class LossMetric:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.average = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.average = self.sum / self.count


# 载入模型
if checkpoint is not None:
    model = torch.load(model_dir / checkpoint)
    print("加载模型:", checkpoint)
else:
    model = AlbertForMultipleChoice.from_pretrained('albert-base-v2')
model.to(device)

loss = LossMetric()
train_dataset = dataProcess.data['test']
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataProcess.collate_fn())
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# 开始训练
model.train()
for epoch in range(epochs):
    tk = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0, leave=True)
    for idx, batch_data in tk:
        optimizer.zero_grad()
        input_ids, attention_mask, token_type_ids, label = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss


def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for idx, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
        optim.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss

        if idx % 20 == 0:
            with torch.no_grad():
                print((outputs[1].argmax(1).data == labels.data).float().mean().item(), loss.item())

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        iter_num += 1
        if (iter_num % 100 == 0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
                epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))
