import os, re
import torch
from torch import nn
import torch.nn.functional as F
import esm
import numpy as np
from Bio import SeqIO


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
        return self.model(x)

def parse_true_label(id_str: str) -> int:
    """从序列ID末尾解析连续数字作为标签"""
    m = re.search(r'(\d+)$', id_str)
    if not m:
        raise ValueError(f"无法从ID解析标签: {id_str}")
    return int(m.group(1))

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型路径
model_path = 'model_pt_files/best_model.pt'

# 兼容两种保存方式：整个模型对象 或 state_dict
loaded = torch.load(model_path, map_location=device)
if isinstance(loaded, nn.Module):
    net = loaded.to(device).eval()
elif isinstance(loaded, dict) and (
    "model_state" in loaded or all(isinstance(v, torch.Tensor) for v in loaded.values())
):
    net = P450HGT().to(device)
    state = loaded.get("model_state", loaded)
    net.load_state_dict(state)
    net.eval()
else:
    raise TypeError("无法识别的模型文件格式，请确认是保存的整个模型或 state_dict。")

# 数据路径
FASTA_PATH = "P450_All_Right_60_40_del_star_with_label.fasta"
EMB_PATH   = "P450_All_Right_60_40_del_star_with_label/"
output_fasta = "P450_All_Right_60_40_class_false.fasta"
os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

EMB_LAYER = 33

# 读取FASTA，并加载对应的嵌入
ids, Xs = [], []
missing = []
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    fn = os.path.join(EMB_PATH, f"{header}.pt")
    try:
        embs = torch.load(fn, map_location="cpu")
        Xs.append(embs['mean_representations'][EMB_LAYER].float())
        ids.append(header)
    except FileNotFoundError:
        missing.append(header)

if missing:
    print(f"[警告] 缺少 {len(missing)} 个嵌入，将跳过。例如：{missing[:3]} ...")

if not Xs:
    raise RuntimeError("没有可用的嵌入向量，请检查 EMB_PATH 与文件命名。")

Xs = torch.stack(Xs, dim=0)

# 分批推理
def batched_predict(model, X, batch_size=4096):
    preds = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            x = X[i:i+batch_size].to(device, non_blocking=True)
            logits = model(x)         
            pred = logits.argmax(dim=1).cpu().tolist()
            preds.extend(pred)
    return preds

preds = batched_predict(net, Xs, batch_size=4096)

# 找出判错的ID及其预测标签
pre_mistake_labels = []
pre_mistake_labels_true_false = {}
for seq_id, y_pred in zip(ids, preds):
    y_true = parse_true_label(seq_id)
    if y_pred != y_true:
        pre_mistake_labels.append(seq_id)
        pre_mistake_labels_true_false[seq_id] = str(y_pred)

print(len(pre_mistake_labels))
print(pre_mistake_labels_true_false)

# 输出判错样本到新FASTA
out_labels, out_seqs = [], []
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    rid = record.id.split()[0]  # 与 read_fasta 的 header 对齐
    if rid in pre_mistake_labels:
        out_labels.append(f"{rid}_false_{pre_mistake_labels_true_false[rid]}")
        out_seqs.append(str(record.seq))

print(len(out_labels))
print(len(out_seqs))

with open(output_fasta, "w") as fw:
    for lab, seq in zip(out_labels, out_seqs):
        fw.write(f">{lab}\n{seq}\n")
