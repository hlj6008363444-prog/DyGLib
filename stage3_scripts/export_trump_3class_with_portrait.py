import os
import argparse
import numpy as np
import pandas as pd


EDGE_TYPES = ["reply", "repost", "quote"]


def one_hot_edge_type(edge_type: str) -> np.ndarray:
    v = np.zeros(len(EDGE_TYPES), dtype=np.float32)
    if edge_type in EDGE_TYPES:
        v[EDGE_TYPES.index(edge_type)] = 1.0
    return v


def build_node_features_from_portrait(
    portrait_csv: str,
    labeled_user_ids: np.ndarray,
    use_window: str = "w5",
) -> tuple[np.ndarray, dict]:
    """
    Build node feature matrix:
      - node_id starts from 1
      - row 0 is all zeros padding
    We aggregate/choose one window per user (default w5).
    """

    df = pd.read_csv(portrait_csv)

    # --- pick one window per user ---
    # expected columns from your file:
    # user_id, window_id, ... , cognitive_resilience, openness, conscientiousness, extraversion, agreeableness, neuroticism, InfluenceScore, ...
    required_cols = [
        "user_id",
        "window_id",
        "cognitive_resilience",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "InfluenceScore",
        "received_like",
        "received_quote",
        "received_repost",
        "total_received_interactions",
        "num_posts_in_window",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[portrait] missing columns: {missing}\n"
            f"Existing columns: {list(df.columns)}"
        )

    # keep only labeled users (optional but cleaner)
    df = df[df["user_id"].isin(labeled_user_ids)].copy()

    # choose a window (w5) per user; fallback to last available window if missing
    df["window_id"] = df["window_id"].astype(str)
    w_df = df[df["window_id"] == use_window].copy()
    if len(w_df) == 0:
        raise ValueError(f"[portrait] no rows found for window_id={use_window}")

    # Some users might not have w5;补齐：对缺失用户取其最大 window_id 的那条
    have = set(w_df["user_id"].unique().tolist())
    need = set(labeled_user_ids.tolist()) - have
    if need:
        df_need = df[df["user_id"].isin(list(need))].copy()
        # window_id like w1..w5 -> sort by numeric suffix
        df_need["_w"] = df_need["window_id"].str.replace("w", "", regex=False).astype(int)
        df_need = df_need.sort_values(["user_id", "_w"]).groupby("user_id").tail(1)
        w_df = pd.concat([w_df, df_need.drop(columns=["_w"])], ignore_index=True)

    # final per-user row
    w_df = w_df.sort_values("user_id").groupby("user_id").tail(1)

    # --- build mapping user_id -> node_id (1..N) ---
    labeled_user_ids_sorted = np.array(sorted(labeled_user_ids.tolist()), dtype=int)
    user_to_node = {uid: idx + 1 for idx, uid in enumerate(labeled_user_ids_sorted)}
    num_nodes = len(labeled_user_ids_sorted)

    # --- features (you can add/remove columns here) ---
    feat_cols = [
        "cognitive_resilience",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "InfluenceScore",
        "received_like",
        "received_quote",
        "received_repost",
        "total_received_interactions",
        "num_posts_in_window",
    ]

    # fill missing to 0
    w_df[feat_cols] = w_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    feat_dim = len(feat_cols)
    node_feats = np.zeros((num_nodes + 1, feat_dim), dtype=np.float32)  # +1 for padding row 0

    for _, row in w_df.iterrows():
        uid = int(row["user_id"])
        if uid not in user_to_node:
            continue
        nid = user_to_node[uid]
        node_feats[nid] = row[feat_cols].to_numpy(dtype=np.float32)

    return node_feats, user_to_node


def build_trump_labels(stance_csv: str, labeled_user_ids: np.ndarray) -> dict:
    df = pd.read_csv(stance_csv)

    required_cols = ["UserId", "target", "stance_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[stance] missing columns: {missing}\n"
            f"Existing columns: {list(df.columns)}"
        )

    df["target"] = df["target"].astype(str).str.lower()
    df = df[df["target"] == "trump"].copy()

    df = df[df["UserId"].isin(labeled_user_ids)].copy()

    # stance_id should be 0/1/2
    df["stance_id"] = pd.to_numeric(df["stance_id"], errors="coerce")
    df = df.dropna(subset=["stance_id"])
    df["stance_id"] = df["stance_id"].astype(int)

    label_map = {int(r["UserId"]): int(r["stance_id"]) for _, r in df.iterrows()}

    # sanity check
    missing_users = set(labeled_user_ids.tolist()) - set(label_map.keys())
    if missing_users:
        # 允许存在少量缺失，但建议你补齐；这里先报 warning
        print(f"[WARN] missing Trump labels for {len(missing_users)} users. Example: {list(sorted(missing_users))[:10]}")

    return label_map


def export_dyglib(
    dynamic_edges_csv: str,
    stance_csv: str,
    portrait_csv: str,
    out_root: str,
    dataset_name: str = "politisky_individual_trump_3class",
    use_window: str = "w5",
    ts_col: str = "ts",
    u_col: str = "u",
    i_col: str = "i",
    edge_type_col: str = "edge_type",
):
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Decide labeled user set (from stance file Trump subset)
    stance_df = pd.read_csv(stance_csv)
    stance_df["target"] = stance_df["target"].astype(str).str.lower()
    trump_users = stance_df[stance_df["target"] == "trump"]["UserId"].dropna().astype(int).unique()
    trump_users = np.array(sorted(trump_users.tolist()), dtype=int)

    # 2) Node features + mapping
    node_feats, user_to_node = build_node_features_from_portrait(
        portrait_csv=portrait_csv,
        labeled_user_ids=trump_users,
        use_window=use_window,
    )

    # 3) Labels map (UserId -> stance_id)
    label_map = build_trump_labels(stance_csv=stance_csv, labeled_user_ids=trump_users)

    # 4) Load dynamic edges
    e = pd.read_csv(dynamic_edges_csv)

    # check required cols
    for c in [u_col, i_col, ts_col, edge_type_col]:
        if c not in e.columns:
            raise ValueError(
                f"[edges] missing column {c} in {dynamic_edges_csv}\n"
                f"Existing columns: {list(e.columns)}"
            )

    # keep only edges between labeled users (first version)
    e[u_col] = pd.to_numeric(e[u_col], errors="coerce")
    e[i_col] = pd.to_numeric(e[i_col], errors="coerce")
    e[ts_col] = pd.to_numeric(e[ts_col], errors="coerce")
    e = e.dropna(subset=[u_col, i_col, ts_col]).copy()
    e[u_col] = e[u_col].astype(int)
    e[i_col] = e[i_col].astype(int)

    e = e[e[u_col].isin(trump_users) & e[i_col].isin(trump_users)].copy()

    # normalize edge_type
    e[edge_type_col] = e[edge_type_col].astype(str).str.lower()
    e = e[e[edge_type_col].isin(EDGE_TYPES)].copy()

    # sort by time (DyGLib expects temporal order)
    e = e.sort_values(ts_col).reset_index(drop=True)

    # 5) Build DyGLib ml_{dataset}.csv with columns u,i,ts,label,idx
    # idx starts at 1; 0 reserved for padding
    num_edges = len(e)
    if num_edges == 0:
        raise ValueError("[edges] no edges left after filtering. Check your dynamic_edges_csv content.")

    ml_rows = []
    edge_feats = np.zeros((num_edges + 1, len(EDGE_TYPES)), dtype=np.float32)  # +1 padding row 0

    for k in range(num_edges):
        row = e.iloc[k]
        src_uid = int(row[u_col])
        dst_uid = int(row[i_col])
        ts = float(row[ts_col])
        et = str(row[edge_type_col]).lower()

        src_nid = user_to_node[src_uid]
        dst_nid = user_to_node[dst_uid]

        # label = stance_id of source user for Trump task
        # if missing label -> skip (or set to 2 Neither); here we skip to keep training clean
        if src_uid not in label_map:
            continue
        y = int(label_map[src_uid])

        idx = len(ml_rows) + 1
        ml_rows.append((src_nid, dst_nid, ts, y, idx))
        edge_feats[idx] = one_hot_edge_type(et)

    if len(ml_rows) == 0:
        raise ValueError("[edges] after removing missing-label edges, no rows remain. Check stance labels coverage.")

    ml_df = pd.DataFrame(ml_rows, columns=["u", "i", "ts", "label", "idx"])

    # 6) Save outputs
    csv_path = os.path.join(out_dir, f"ml_{dataset_name}.csv")
    edge_npy_path = os.path.join(out_dir, f"ml_{dataset_name}.npy")
    node_npy_path = os.path.join(out_dir, f"ml_{dataset_name}_node.npy")

    ml_df.to_csv(csv_path, index=False)
    np.save(edge_npy_path, edge_feats[: (len(ml_rows) + 1)])
    np.save(node_npy_path, node_feats)

    print("=== Export finished ===")
    print(f"Dataset dir: {out_dir}")
    print(f"Edges csv:   {csv_path}  (rows={len(ml_df)})")
    print(f"Edge feats:  {edge_npy_path}  shape={edge_feats[: (len(ml_rows) + 1)].shape}")
    print(f"Node feats:  {node_npy_path}  shape={node_feats.shape}")
    print(f"Num nodes:   {node_feats.shape[0]-1} (padding row 0 reserved)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamic_edges_csv", type=str, required=True, help="CSV with u,i,ts,edge_type (user ids)")
    ap.add_argument("--stance_csv", type=str, required=True, help="user_stance_labels.csv")
    ap.add_argument("--portrait_csv", type=str, required=True, help="user_profiles_complete.csv")
    ap.add_argument("--out_root", type=str, required=True, help="DyGLib processed_data root, e.g. D:\\...\\DyGLib\\processed_data")
    ap.add_argument("--dataset_name", type=str, default="politisky_individual_trump_3class")
    ap.add_argument("--use_window", type=str, default="w5", help="which window_id to use for node portrait features")
    ap.add_argument("--u_col", type=str, default="u")
    ap.add_argument("--i_col", type=str, default="i")
    ap.add_argument("--ts_col", type=str, default="ts")
    ap.add_argument("--edge_type_col", type=str, default="edge_type")
    args = ap.parse_args()

    export_dyglib(
        dynamic_edges_csv=args.dynamic_edges_csv,
        stance_csv=args.stance_csv,
        portrait_csv=args.portrait_csv,
        out_root=args.out_root,
        dataset_name=args.dataset_name,
        use_window=args.use_window,
        u_col=args.u_col,
        i_col=args.i_col,
        ts_col=args.ts_col,
        edge_type_col=args.edge_type_col,
    )


if __name__ == "__main__":
    main()