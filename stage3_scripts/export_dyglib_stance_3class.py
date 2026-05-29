import os
import numpy as np
import pandas as pd
import argparse

EDGE_FEAT_DIM = 172
NODE_FEAT_DIM = 172


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def export_stance_3class(
    edges_csv_path: str,
    stance_labels_csv_path: str,
    dyglib_processed_data_dir: str,
    target: str,
    dataset_name: str,
):
    dataset_dir = os.path.join(dyglib_processed_data_dir, dataset_name)
    ensure_dir(dataset_dir)

    out_csv = os.path.join(dataset_dir, f"ml_{dataset_name}.csv")
    out_edge_npy = os.path.join(dataset_dir, f"ml_{dataset_name}.npy")
    out_node_npy = os.path.join(dataset_dir, f"ml_{dataset_name}_node.npy")
    out_map_csv = os.path.join(dataset_dir, "user_id_map.csv")

    print("Loading edges:", edges_csv_path)
    edges = pd.read_csv(edges_csv_path, encoding="utf-8-sig")

    # 期望列：u,i,ts,edge_type
    required = ["u", "i", "ts", "edge_type"]
    miss = [c for c in required if c not in edges.columns]
    if miss:
        raise KeyError(f"edges missing columns {miss}, got {list(edges.columns)}")

    edges = edges.dropna(subset=["u", "i", "ts"]).copy()
    edges["u"] = edges["u"].astype(int)
    edges["i"] = edges["i"].astype(int)
    edges["ts"] = edges["ts"].astype(float)
    edges["edge_type"] = edges["edge_type"].astype(str)

    print("Loading stance labels:", stance_labels_csv_path)
    st = pd.read_csv(stance_labels_csv_path, encoding="utf-8-sig")

    # 标准化 target 字段（兼容 Trump/Harris 大小写）
    st["target_norm"] = st["target"].astype(str).str.strip().str.lower()
    target_norm = target.strip().lower()
    st_t = st[st["target_norm"] == target_norm].copy()

    if len(st_t) == 0:
        raise ValueError(f"No labels found for target={target}. Unique targets: {sorted(st['target_norm'].unique().tolist())}")

    # 三分类 label：直接用 stance_id (期望 0/1/2)
    # Favor->0, Against->1, Neither->2 （按你的指引）
    if "stance_id" not in st_t.columns:
        raise KeyError("stance_labels_csv missing stance_id column")

    st_t["stance_id"] = st_t["stance_id"].astype(int)
    bad = st_t[~st_t["stance_id"].isin([0, 1, 2])]
    if len(bad) > 0:
        raise ValueError(f"Found stance_id not in [0,1,2], sample:\n{bad.head(5)}")

    # 建立 node 映射：保证 edges 出现的所有 u/i 和 stance 出现的 UserId 都能映射
    all_users = pd.Index(
        pd.concat([edges["u"], edges["i"], st_t["UserId"]], ignore_index=True)
        .dropna()
        .astype(int)
        .unique()
    )
    all_users = all_users.sort_values()
    node_ids = np.arange(1, len(all_users) + 1, dtype=np.int64)

    user2node = dict(zip(all_users.tolist(), node_ids.tolist()))
    node2user = dict(zip(node_ids.tolist(), all_users.tolist()))

    pd.DataFrame({"UserId": all_users.astype(int), "node_id": node_ids}).to_csv(out_map_csv, index=False, encoding="utf-8-sig")
    print("Saved user_id_map:", out_map_csv, "rows:", len(all_users))

    # 映射边的 u/i 为 node_id（DyGLib 要求）
    edges["u"] = edges["u"].map(user2node).astype(int)
    edges["i"] = edges["i"].map(user2node).astype(int)

    # label：用源节点 u 对应的 UserId 去查 stance_id（缺失先填 2=Neither，避免默认为 Against 之类）
    label_map = dict(zip(st_t["UserId"].astype(int).tolist(), st_t["stance_id"].astype(int).tolist()))
    def node_u_to_label(node_u: int) -> int:
        uid = node2user.get(int(node_u), -1)
        return int(label_map.get(uid, 2))  # default Neither

    edges["label"] = edges["u"].map(node_u_to_label).astype(int)

    # idx 从 1 开始
    edges = edges.reset_index(drop=True)
    edges["idx"] = np.arange(1, len(edges) + 1, dtype=np.int64)

    ml = edges[["u", "i", "ts", "label", "idx"]].copy()
    ml.to_csv(out_csv, index=False)
    print("Saved:", out_csv, "rows:", len(ml))
    print("label distribution (by events):")
    print(ml["label"].value_counts(dropna=False).sort_index())

    # 边特征：onehot(repost/quote/reply) + weight，再 padding 到 172
    et = edges["edge_type"]
    w = et.map({"repost": 3, "quote": 4, "reply": 2}).fillna(1).to_numpy().reshape(-1, 1)

    onehot = np.zeros((len(edges), 3), dtype=np.float32)
    onehot[:, 0] = (et == "repost").to_numpy(dtype=np.float32)
    onehot[:, 1] = (et == "quote").to_numpy(dtype=np.float32)
    onehot[:, 2] = (et == "reply").to_numpy(dtype=np.float32)

    small = np.concatenate([onehot, w.astype(np.float32)], axis=1)  # (E, 4)

    edge_feat = np.zeros((len(edges) + 1, EDGE_FEAT_DIM), dtype=np.float32)  # padding row 0
    edge_feat[1:, :small.shape[1]] = small
    np.save(out_edge_npy, edge_feat)
    print("Saved:", out_edge_npy, "shape:", edge_feat.shape)

    # 节点特征：先全 0（后续再加画像/结构特征）
    node_feat = np.zeros((len(all_users) + 1, NODE_FEAT_DIM), dtype=np.float32)  # padding row 0
    np.save(out_node_npy, node_feat)
    print("Saved:", out_node_npy, "shape:", node_feat.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--dyglib_dir", required=True)
    ap.add_argument("--target", required=True, choices=["Trump", "Harris", "trump", "harris"])
    ap.add_argument("--dataset_name", required=True)
    args = ap.parse_args()

    export_stance_3class(
        edges_csv_path=args.edges_csv,
        stance_labels_csv_path=args.labels_csv,
        dyglib_processed_data_dir=args.dyglib_dir,
        target=args.target,
        dataset_name=args.dataset_name,
    )