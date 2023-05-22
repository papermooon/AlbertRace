import torch
from tqdm import tqdm
from transformers import AlbertForMultipleChoice
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
import dataProcess
from torch.utils.tensorboard import SummaryWriter
from transformers import logging
from transformers import get_linear_schedule_with_warmup

logging.set_verbosity_error()

# 超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = Path("./model")
checkpoint = 'epoch_4_05-19_09-57.pt'
# checkpoint = None
batch_size = 14
epochs = 5


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
    model = AlbertForMultipleChoice.from_pretrained('albert-large-v2')
model.to(device)

# loss = LossMetric()
# writer = SummaryWriter(log_dir='./logs')
writer = SummaryWriter(log_dir='/root/tf-logs')
train_dataset = dataProcess.RaceDataset(dataProcess.data['train'])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataProcess.collate_fn,
                              num_workers=8)
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epochs)

current_step = 0
# 梯度累积
gradient_accumulation_steps = 4
# 开始训练
model.train()
for epoch in range(epochs):
    epoch_ct = 0
    current_loss = 0
    tk = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0, leave=True)
    for idx, batch_data in tk:
        # optimizer.zero_grad()

        input_ids, attention_mask, token_type_ids, labels = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        token_type_ids = token_type_ids.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss

        current_loss = current_loss + loss.item()

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if current_step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # optimizer.step()
        # scheduler.step()

        writer.add_scalar(tag="batch_loss_large",  # 可以理解为图像的名字
                          scalar_value=loss.item()*gradient_accumulation_steps,  # 纵坐标的值
                          global_step=current_step  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        current_step = current_step + 1
        epoch_ct = epoch_ct + 1

        avg_loss = current_loss / epoch_ct

        tk.set_description("Epoch {}/{}".format(epoch + 1, epochs))
        tk.set_postfix(loss=loss.item()*gradient_accumulation_steps, loss_avg=avg_loss)

    now_time = datetime.now().strftime('%m-%d_%H-%M')
    save_name = "epoch_" + str(epoch) + "_" + now_time + ".pt"
    torch.save(model, model_dir / save_name)
