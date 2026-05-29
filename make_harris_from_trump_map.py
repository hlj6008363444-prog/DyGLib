import os
import pandas as pd
import shutil

TRUMP_DIR = r"processed_data/politisky_individual_trump"
HARRIS_DIR = r"processed_data/politisky_individual_harris"
STANCE_LABELS = r"D:\毕业论文\new_project\code\reproduce\data\user_stance_labels.csv"

trump_ml = os.path.join(TRUMP_DIR, "ml_politisky_individual_trump.csv")
trump_map = os.path.join(TRUMP_DIR, "user_id_map.csv")

harris_ml = os.path.join(HARRIS_DIR, "ml_politisky_individual_harris.csv")

# 1) 读 trump 边（里面的 u/i 已经是 node_id）
edges = pd.read_csv(trump_ml)

# 2) 读 node_id <-> UserId 映射（你脚本保存的是 UserId,node_id）
m = pd.read_csv(trump_map)
node2user = dict(zip(m["node_id"].astype(int), m["UserId"].astype(int)))

# 3) 读 stance labels（UserId,target,stance,stance_id,...）
st = pd.read_csv(STANCE_LABELS, encoding="utf-8-sig")
st["target"] = st["target"].astype(str).str.lower()

# 4) Harris 二分类：Favor=1，否则0（保持与你 trump_binary 脚本一致的定义）
st_h = st[st["target"] == "harris"].copy()
st_h["label_bin"] = (st_h["stance"].astype(str) == "Favor").astype(int)
user2label = dict(zip(st_h["UserId"].astype(int), st_h["label_bin"].astype(int)))

# 5) 把每条边样本的 label 替换为 “源节点u对应的UserId的Harris标签”
def u_to_harris_label(u_node_id: int) -> int:
    uid = node2user.get(int(u_node_id), None)
    if uid is None:
        return 0
    return int(user2label.get(uid, 0))

edges["label"] = edges["u"].map(u_to_harris_label).astype(int)

# 6) 输出 Harris ml csv（u,i,ts,label,idx）
os.makedirs(HARRIS_DIR, exist_ok=True)
edges.to_csv(harris_ml, index=False)
print("wrote:", harris_ml, "rows:", len(edges))

# 7) 复制/改名 npy（特征不变，文件名要匹配 dataset_name）
for suffix in [".npy", "_node.npy"]:
    src = os.path.join(TRUMP_DIR, f"ml_politisky_individual_trump{suffix}")
    dst = os.path.join(HARRIS_DIR, f"ml_politisky_individual_harris{suffix}")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("copied:", dst)
    else:
        print("WARN missing:", src)