import torch
from tqdm import tqdm
from transformers import AlbertForMultipleChoice
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
import dataProcess
from torch.utils.tensorboard import SummaryWriter

# 超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = Path("./model")
checkpoint = None
batch_size = 64
epochs = 20


# loss的进度条,暂时弃用
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
writer = SummaryWriter(log_dir='./logs')
train_dataset = dataProcess.RaceDataset(dataProcess.data['test'])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataProcess.collate_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
current_step = 0

# 开始训练
model.train()
for epoch in range(epochs):
    tk = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0, leave=True)
    for idx, batch_data in tk:
        optimizer.zero_grad()

        input_ids, attention_mask, token_type_ids, labels = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        writer.add_scalar(tag="batch_loss",  # 可以理解为图像的名字
                          scalar_value=loss.item(),  # 纵坐标的值
                          global_step=current_step  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        current_step = current_step + 1

        tk.set_description("Epoch {}/{}".format(epoch + 1, epochs))
        tk.set_postfix(loss=loss.item())

    now_time = datetime.now().strftime('%m-%d_%H-%M')
    save_name = "epoch_" + str(epoch) + "_" + now_time + ".pt"
    torch.save(model, model_dir / save_name)



