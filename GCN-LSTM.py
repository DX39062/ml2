# ----------------------------------------------------------------------------
# 完整脚本：用于fMRI时空数据分类的GCN-LSTM模型
# 架构：PyTorch, 5折分层交叉验证, CUDA加速
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd  # <-- 已添加
import os          # <-- 已添加

# ---------------------------------
# 第一部分：全局配置与设备设置
# ---------------------------------

# --- 超参数（遵循模板 ）---
K_FOLDS = 5
N_EPOCHS = 100  # 中为100
BATCH_SIZE = 1    # 中为1
LEARNING_RATE = 0.0001 # 中为0.0001
RANDOM_SEED = 42

# --- 数据维度（根据用户查询）---
N_NODES = 116       # 116个脑区 (ROIs)
N_TIME_STEPS = 140  # 140个时间点
N_CLASSES = 2       # 二元分类 (例如：HC vs EMCI )

# --- GCN-LSTM 模型参数 ---
GCN_HIDDEN_DIM = 32
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
DROPOUT_RATE = 0.5

# --- 设置随机种子以保证可复现性 ---
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# --- 1.1 CUDA设备配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- 报告：GCN-LSTM模型 ---")
print(f"--- 运行设备: {device} ---")


# ---------------------------------
# 第二部分：图数据预处理工具
# ---------------------------------

def normalize_adjacency_matrix(A):
    """
    计算对称归一化的邻接矩阵 A_hat = D^(-1/2) * (A + I) * D^(-1/2)
    这是Kipf & Welling (2017) GCN的经典实现 [22, 23]。
    
    参数:
    A (torch.Tensor): 邻接矩阵 (N, N)
    
    返回:
    torch.Tensor: 归一化后的邻接矩阵 A_hat (N, N)
    """
    # 1. 创建 A_tilde = A + I
    A_tilde = A + torch.eye(A.shape[0], device=A.device)
    
    # 2. 计算度矩阵 D_tilde 的逆平方根
    degrees = torch.sum(A_tilde, dim=1)
    D_inv_sqrt = torch.pow(degrees, -0.5)
    
    # 3. 处理孤立节点（度为0），防止出现inf 
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    
    # 4. 创建对角矩阵 D_inv_sqrt_matrix
    D_inv_sqrt_matrix = torch.diag(D_inv_sqrt)
    
    # 5. 计算 A_hat = D_matrix @ A_tilde @ D_matrix
    A_hat = D_inv_sqrt_matrix @ A_tilde @ D_inv_sqrt_matrix
    return A_hat


# ---------------------------------
# 第三部分：核心模型架构
# ---------------------------------

class GCN_LSTM_Classifier(nn.Module):
    """
    GCN-LSTM 时空分类器
    该模型首先在每个时间点上应用GCN提取空间特征 ，
    然后使用LSTM学习这些空间特征随时间演变的模式 [24, 1, 25]。
    """
    def __init__(self, in_features, gcn_hidden, lstm_hidden, num_lstm_layers, n_classes, dropout):
        super(GCN_LSTM_Classifier, self).__init__()
        
        self.n_nodes = N_NODES
        self.n_time_steps = N_TIME_STEPS
        self.gcn_hidden_dim = gcn_hidden
        self.lstm_hidden_dim = lstm_hidden
        
        # 3.2.1 GCN层：
        # 我们将GCN实现为 (A_hat @ X @ W)
        # nn.Linear(in_features, gcn_hidden) 实现了 (X @ W) 部分 [26]
        # 'in_features' 在此为1，因为每个节点在每个时间点的输入是一个标量。
        self.gcn_layer = nn.Linear(in_features, gcn_hidden)
        
        # 3.2.2 LSTM层：
        # input_size 是GCN层的输出维度
        # 我们将为116个节点中的每一个并行运行一个LSTM（权重共享）
        self.lstm_layer = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,  # 接受 (Batch, Seq, Features)
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # 3.2.3 分类头：
        # 我们将116个节点的最终LSTM隐藏状态全部连接起来
        self.classifier_head = nn.Linear(
            in_features=lstm_hidden * N_NODES,
            out_features=n_classes
        )
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, adj_hat):
        """
        前向传播
        x 形状: (Batch, T=140, N=116)
        adj_hat 形状: (N=116, N=116)
        """
        B = x.shape[0] # Batch size
        
        # --- 1. GCN 空间特征提取 (在每个时间点上) ---
        
        # a. 准备 GCN 输入：(B, T, N) -> (B, T, N, F_in=1)
        x = x.unsqueeze(-1)
        
        # b. GCN操作: H' = sigma(A_hat @ H @ W)
        # H @ W (X @ W): 使用nn.Linear高效实现
        gcn_out = self.gcn_layer(x)  # (B, T, N, F_in=1) -> (B, T, N, F_gcn)
        
        # A_hat @ (H @ W): 使用einsum进行高效的批处理矩阵乘法
        # 'nn' (A_hat) @ 'btni' (gcn_out) -> 'btni'
        gcn_out = torch.einsum('nn,btni->btni', adj_hat, gcn_out)
        gcn_out = F.relu(gcn_out)
        
        # gcn_out 形状: (B, T=140, N=116, F_gcn)

        # --- 2. LSTM 时间特征提取 ---
        
        # a. 准备 LSTM 输入：
        # 我们希望为116个节点中的每一个都运行一个LSTM
        # 交换 T 和 N 维度: (B, T, N, F) -> (B, N, T, F)
        lstm_in = gcn_out.permute(0, 2, 1, 3)
        
        # b. Reshape 以进行并行处理
        # (B, N, T, F) -> (B * N, T, F)
        lstm_in = lstm_in.reshape(-1, self.n_time_steps, self.gcn_hidden_dim)
        
        # c. 运行 LSTM
        # 我们只关心最后一个时间步的隐藏状态 h_n
        # h_n 形状: (Num_layers, B * N, F_lstm)
        _, (h_n, _) = self.lstm_layer(lstm_in)
        
        # d. 获取最后一层的隐藏状态
        lstm_out = h_n[-1]  # 形状: (B * N, F_lstm)
        
        # --- 3. 分类 (Readout) ---
        
        # a. Reshape: 将116个节点的特征重新组合
        # (B * N, F_lstm) -> (B, N * F_lstm)
        classifier_in = lstm_out.reshape(B, -1)
        classifier_in = self.dropout_layer(classifier_in)
        
        # b. 最终分类
        output = self.classifier_head(classifier_in)  # 形状: (B, N_CLASSES)
        
        return output

# ---------------------------------
# 第四部分：数据封装
# ---------------------------------

class fMRISpatioTemporalDataset(Dataset):
    """
    自定义PyTorch数据集，用于加载fMRI时空数据
    """
    def __init__(self, X_data, y_data):
        # X_data 应为 (NumSubjects, T=140, N=116)
        # y_data 应为 (NumSubjects,)
        self.X = X_data
        self.y = y_data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------
# 第五部分：5折交叉验证与训练主循环
# ---------------------------------

def train_epoch(model, loader, criterion, optimizer, adj_hat, device):
    """ 单个epoch的训练循环 """
    model.train()  # 设置为训练模式 
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播 (adj_hat 在GPU上是固定的)
        outputs = model(X_batch, adj_hat)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        
        # 存储预测和标签以计算BA
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, adj_hat, device):
    """ 单个epoch的验证循环 """
    model.eval()  # 设置为评估模式 
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch, adj_hat)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item() * X_batch.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    # 遵循，使用平衡准确率 
    epoch_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds

# --- 5.1 加载和准备数据 ---
print("\n--- 正在加载和准备数据 (CSV)... ---")

# --- 5.1.a 加载 邻接矩阵 (A) ---
A_data_file = 'FC.csv'  # <-- 已根据您的请求更新
# 假设 FC.csv 没有表头
try:
    A_data_np = pd.read_csv(A_data_file, header=None).values
    print(f"邻接 (A) 形状: {A_data_np.shape}")
    if A_data_np.shape!= (N_NODES, N_NODES):
        print(f"警告：邻接矩阵形状 {A_data_np.shape} 与 N_NODES={N_NODES} 不匹配。")
except FileNotFoundError:
    print(f"错误：找不到文件 '{A_data_file}'。请确保文件在脚本目录中。")
    exit()

# --- 5.1.b 加载并解析 标签 (y) ---
y_data_file = 'labels.csv' # <-- 已根据您的请求更新
label_map = {}
try:
    # 1. 将 'labels.csv' 作为以':'分隔的文件加载
    #    这匹配了您图片中的 "'id': label," 格式
    label_df = pd.read_csv(y_data_file, sep=':', header=None, names=['id', 'label'])
    
    # 2. 清理 'id' 列 (去除空格和单引号)
    id_col = label_df['id'].str.strip().str.strip("'")
    
    # 3. 清理 'label' 列 (去除空格、尾随逗号，并转为整数)
    label_col = label_df['label'].str.strip().str.rstrip(',').astype(int)
    
    # 4. 创建一个 {id -> label} 的查找字典
    label_map = pd.Series(label_col.values, index=id_col.values).to_dict()
    
    print(f"成功加载并解析了 {len(label_map)} 个标签。")
    
except FileNotFoundError:
    print(f"错误：找不到文件 '{y_data_file}'。请确保文件在脚本目录中。")
    exit()
except Exception as e:
    print(f"错误：解析 '{y_data_file}' 时出错。请确保其格式为 'id': label, (如 '005_S_0602': 0,)")
    print(f"Pandas 错误: {e}")
    exit()

# --- 5.1.c 加载并匹配 时序特征 (X) ---
X_data_folder = 'tsdatasets' # <-- 已根据您的请求更新
all_subjects_ts = []
all_labels_list = []

try:
    # 获取文件夹中所有.csv 文件
    subject_files = sorted([f for f in os.listdir(X_data_folder) if f.endswith('.csv')])
except FileNotFoundError:
    print(f"错误：找不到文件夹 '{X_data_folder}'。请确保文件夹在脚本目录中。")
    exit()

print(f"正在从 {len(subject_files)} 个CSV文件中加载并匹配时序数据...")
scaler = StandardScaler()
files_matched = 0

for filename in subject_files:
    # 1. 从文件名中提取ID (例如 '129_S_6228' 来自 '129_S_6228_20180220.csv')
    file_id = '_'.join(filename.split('_')[:3])
    
    # 2. 在标签字典中查找该ID
    label = label_map.get(file_id)
    
    # 3. 如果找到匹配项，则加载数据
    if label is not None:
        files_matched += 1
        filepath = os.path.join(X_data_folder, filename)
        timeseries_data = pd.read_csv(filepath, header=None).values # 形状 (140, 116)
        
        if timeseries_data.shape!= (N_TIME_STEPS, N_NODES):
             print(f"  > 警告：文件 {filename} 的形状是 {timeseries_data.shape}，"
                   f"但预期是 ({N_TIME_STEPS}, {N_NODES})。")
        
        # 按照脚本中的原始逻辑，对每个受试者的时序数据进行标准化
        timeseries_data_scaled = scaler.fit_transform(timeseries_data)
        
        all_subjects_ts.append(timeseries_data_scaled)
        all_labels_list.append(label)
    else:
        # 如果在标签字典中找不到ID，则打印警告
        print(f"  > 警告: 找不到 '{filename}' (提取的ID: {file_id}) 的标签。将跳过此文件。")

print(f"加载完成。总共匹配了 {files_matched} 个文件和标签。")

# --- 5.1.d 转换为 PyTorch 张量 ---
if files_matched == 0:
    print("错误：未找到任何匹配的时序文件和标签。请检查您的ID格式和文件名。")
    exit()
    
# 将匹配的列表转换为Numpy数组
X_data_np = np.stack(all_subjects_ts, axis=0)
y_data_np = np.array(all_labels_list)

print(f"最终特征 (X) 形状: {X_data_np.shape}")
print(f"最终标签 (y) 形状: {y_data_np.shape}")

# 确保邻接矩阵对角线为0 (自环将在归一化函数中添加)
np.fill_diagonal(A_data_np, 0)

# 将数据转换为PyTorch张量
X_data_torch = torch.tensor(X_data_np, dtype=torch.float32)
y_data_torch = torch.tensor(y_data_np, dtype=torch.long)
A_data_torch = torch.tensor(A_data_np, dtype=torch.float32)


# --- 5.1.1 邻接矩阵预处理 ---
# 这是一个关键优化：
# 我们在所有折（fold）和epoch之外计算一次 A_hat [27, 28]
A_hat = normalize_adjacency_matrix(A_data_torch)
A_hat = A_hat.to(device)  # 将 A_hat 一次性传输到GPU
print("邻接矩阵已归一化并发送到GPU。")

# --- 5.1.2 类别不平衡处理  ---
# 计算类别权重以用于损失函数 
class_counts = torch.bincount(y_data_torch)
# 防止某个类别在数据子集中不存在（例如，在非常小的数据集中）
if len(class_counts) < N_CLASSES:
    # 如果缺少类别，则使用临时权重1，但这表明数据/分割有问题
    print(f"警告：标签数据只包含 {len(class_counts)} 个类别，但预期有 {N_CLASSES} 个。")
    class_weights = torch.ones(N_CLASSES, device=device)
else:
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / torch.sum(class_weights) # 归一化

class_weights = class_weights.to(device) # 发送到GPU

print(f"检测到类别不平衡：{class_counts.tolist()}")
print(f"计算出的类别权重：{class_weights.tolist()}")

# --- 5.3 StratifiedKFold 主循环 ---
print(f"\n--- 开始 {K_FOLDS}-折交叉验证... ---")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# 存储每个折的结果
fold_results = []
# 存储所有折的最佳验证混淆矩阵
all_fold_best_cm = []

# 创建完整数据集
full_dataset = fMRISpatioTemporalDataset(X_data_torch, y_data_torch)

start_cv_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_data_torch, y_data_torch)):
    
    print(f"\n--- ---")
    fold_start_time = time.time()
    
    # 5.3.1 为当前折创建 DataLoaders
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    # shuffle=True 仅用于训练集
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5.3.2 重新初始化模型和优化器（关键！）
    # 确保每个折都是从头开始独立训练的
    model = GCN_LSTM_Classifier(
        in_features=1,  # F_in=1
        gcn_hidden=GCN_HIDDEN_DIM,
        lstm_hidden=LSTM_HIDDEN_DIM,
        num_lstm_layers=LSTM_NUM_LAYERS,
        n_classes=N_CLASSES,
        dropout=DROPOUT_RATE
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 定义损失函数（包含类别权重）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 5.3.3 运行 Epochs
    best_val_acc = -1.0
    best_cm = None
    
    for epoch in range(N_EPOCHS):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, A_hat, device)
        val_loss, val_acc, val_labels, val_preds = validate_epoch(model, val_loader, criterion, A_hat, device)
        
        epoch_duration = time.time() - epoch_start_time
        
        if (epoch + 1) % 20 == 0:  # 每20个epoch打印一次
            print(f"  Epoch {epoch+1:3}/{N_EPOCHS} | "
                  f"耗时: {epoch_duration:.2f}s | "
                  f"训练 损失: {train_loss:.4f}, BA: {train_acc:.4f} | "
                  f"验证 损失: {val_loss:.4f}, BA: {val_acc:.4f}")
        
        # 跟踪最佳性能
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 计算并存储最佳混淆矩阵
            best_cm = confusion_matrix(val_labels, val_preds)
            # (可选) 在此处保存最佳模型
            # torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')

    fold_duration = time.time() - fold_start_time
    print(f"--- 折 {fold+1} 完成 (耗时: {fold_duration/60:.2f} 分钟) ---")
    print(f"--- 折 {fold+1} 最佳验证平衡准确率 (BA): {best_val_acc:.4f} ---")
    
    fold_results.append(best_val_acc)
    all_fold_best_cm.append(best_cm)

# --- 5.4 性能聚合与报告 ---
total_cv_time = time.time() - start_cv_time
print(f"\n--- 交叉验证全部完成 (总耗时: {total_cv_time/60:.2f} 分钟) ---")

# 计算并打印最终的平均值和标准差
mean_acc = np.mean(fold_results)
std_acc = np.std(fold_results)

print(f"平均验证平衡准确率 (BA): {mean_acc:.4f}")
print(f"验证平衡准确率标准差 (Std): {std_acc:.4f}")
print("\n--- 5折交叉验证性能汇总表 (表1) ---")

# 打印表格
print("=" * 30)
print(f"| {'折 (Fold)':^8} | {'验证 BA':^18} |")
print("|" + "-" * 8 + "|" + "-" * 18 + "|")
for i, acc in enumerate(fold_results):
    print(f"| {i+1:^8} | {acc:^18.4f} |")
print("=" * 30)
print(f"| {'平均 (Avg)':^8} | {mean_acc:^18.4f} |")
print(f"| {'标准差 (Std)':^8} | {std_acc:^18.4f} |")
print("=" * 30)

# 计算并打印平均混淆矩阵
mean_cm = np.mean(all_fold_best_cm, axis=0)
print("\n平均混淆矩阵 (四舍五入到最近的样本):")
print(np.round(mean_cm))
print("\n--- 脚本执行完毕 ---")