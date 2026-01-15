import json
import numpy as np
from sklearn.cluster import KMeans
import os

def extract_group_colors(input_json="output/clustering_individual/colors.json",
                         output_json="output/clustering_group/group_colors.json",
                         K=16):

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_colors = []
    for _, info in data.items():
        all_colors.extend(info["centers"])

    all_colors = np.array(all_colors)

    # ------------------------------
    # 防止样本数量 < K 导致 KMeans 报错
    # ------------------------------
    K = min(K, len(all_colors))
    if K < 1:
        raise ValueError("二次聚类输入颜色为空，无法继续。")

    km = KMeans(n_clusters=K, random_state=42).fit(all_colors)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"group_centers": km.cluster_centers_.tolist()}, f, indent=2)

    print(f"[K-Means-2] 完成二次聚类，使用 K = {K}。")
