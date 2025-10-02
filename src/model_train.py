# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 创建序列数据集
class MyData(Dataset):
    def __init__(self, root_dir, data_dir):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.file_path = os.path.join(self.root_dir, self.data_dir)
        self.seq_path = os.listdir(self.file_path)

    def __getitem__(self, index):
        seq_path = self.file_path + '/' + str(self.seq_path[index])
        encode_data = torch.load(seq_path)
        name = encode_data['label']
        label = int(seq_path.split(".pt")[0][-1])
        encode_seq = encode_data['mean_representations'][33]
        return label, encode_seq

    def __len__(self):
        return len(self.seq_path)

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 准备训练和测试数据集
root_dir = "/data0/P450_multi_classification_model_data"
train_dir = "train"
test_dir = "test"
train_data = MyData(root_dir, train_dir)
test_data = MyData(root_dir, test_dir)

# 输出训练数据集和测试数据集的大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 Dataloader 来加载数据
train_dataloader = DataLoader(train_data, batch_size=512, drop_last=True, shuffle=True)
# FIX: 测试集不打乱、不丢尾
test_dataloader = DataLoader(test_data, batch_size=512, drop_last=False, shuffle=False)

# 创建神经网络
class P450HGT(nn.Module):
    def __init__(self):
        super(P450HGT, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.model(x)
        # FIX: 不做 softmax，交叉熵内部会做 log_softmax
        return x

# 创建网络模型
hgt = P450HGT()
hgt = hgt.to(device)
print(hgt)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.0001
optimizer = torch.optim.Adam(hgt.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 100

# 添加tensorboard
writer = SummaryWriter("./train_logs_weighted/")
start_time = time.time()

# 确保保存目录存在
os.makedirs("model_pt_files_weighted", exist_ok=True)

# 训练步骤开始
for i in range(epoch):
    print("------------第{}轮开始------------".format(i+1))
    hgt.train()

    total_train_loss = 0.0  # FIX: 累计到样本层面
    total_train_samples = 0
    y_true_train = []
    y_pred_train = []

    pbar_train = tqdm(train_dataloader)
    for labels, encode_seqs in pbar_train:
        labels = labels.to(device).long()          
        encode_seqs = encode_seqs.to(device).float()  

        outputs = hgt(encode_seqs)
        loss = loss_fn(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_train_loss += loss.item() * bs        
        total_train_samples += bs

        y_true_train.extend(labels.tolist())
        y_pred_train.extend(outputs.argmax(dim=1).tolist())

    train_avg_loss = total_train_loss / max(1, total_train_samples)
    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_pre = precision_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
    train_rec = recall_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
    train_f1 = f1_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
    print("epoch: {}, Loss:{:.6f}, acc: {:.4f}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(
        i, train_avg_loss, train_acc, train_pre, train_rec, train_f1))

    writer.add_scalar("train_loss", train_avg_loss, total_train_step)
    writer.add_scalar("train_accuracy", train_acc, total_train_step)
    writer.add_scalar("train_pre", train_pre, total_train_step)
    writer.add_scalar("train_rec", train_rec, total_train_step)
    writer.add_scalar("train_f1", train_f1, total_train_step)
    total_train_step = total_train_step + 1

    # 测试步骤开始
    hgt.eval()
    total_test_loss = 0.0    # FIX: 累计到样本层面
    total_test_samples = 0
    y_true_test = []
    y_pred_test = []

    with torch.no_grad():
        pbar_test = tqdm(test_dataloader)
        for labels, encode_seqs in pbar_test:
            labels = labels.to(device).long()            # FIX
            encode_seqs = encode_seqs.to(device).float()

            outputs = hgt(encode_seqs)
            loss = loss_fn(outputs, labels)

            bs = labels.size(0)
            total_test_loss += loss.item() * bs          # FIX
            total_test_samples += bs

            y_true_test.extend(labels.tolist())
            y_pred_test.extend(outputs.argmax(dim=1).tolist())

        test_avg_loss = total_test_loss / max(1, total_test_samples)  # FIX
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_pre = precision_score(y_true_test, y_pred_test, average='weighted', zero_division=0)
        test_rec = recall_score(y_true_test, y_pred_test, average='weighted', zero_division=0)
        test_f1 = f1_score(y_true_test, y_pred_test, average='weighted', zero_division=0)
        print("epoch: {}, Loss:{:.6f}, acc: {:.4f}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(
            i, test_avg_loss, test_acc, test_pre, test_rec, test_f1))

        writer.add_scalar("test_loss", test_avg_loss, total_test_step)
        writer.add_scalar("test_accuracy", test_acc, total_test_step)
        writer.add_scalar("test_pre", test_pre, total_test_step)
        writer.add_scalar("test_rec", test_rec, total_test_step)
        writer.add_scalar("test_f1", test_f1, total_test_step)
        total_test_step = total_test_step + 1

    # 保存整个模型对象
    torch.save(hgt, "model_pt_files_weighted/{}.pt".format(i))
    print("模型已保存")

writer.close()

