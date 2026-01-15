import os
import cv2
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm  # 进度条
import time

def extract_individual_colors(input_dir="datasets",
                              output_json="output/clustering_individual/colors.json",
                              K=10):

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    results = {}

    img_list = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))]

    start_time = time.time()  # 开始计时

    # 使用 tqdm 显示进度条
    for img_name in tqdm(img_list, desc="Clustering Images", ncols=100):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 仅使用非白背景的像素
        mask = np.any(img_rgb < 240, axis=2)
        pixels = img_rgb[mask].reshape(-1, 3)

        if len(pixels) == 0:
            continue  # 避免空图像导致报错

        km = KMeans(n_clusters=K, random_state=42).fit(pixels)
        centers = km.cluster_centers_
        labels = km.labels_

        _, counts = np.unique(labels, return_counts=True)
        ratios = (counts / counts.sum()).tolist()

        results[img_name] = {
            "centers": centers.tolist(),
            "ratios": ratios
        }

    # 保存 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    end_time = time.time()  # 结束计时
    elapsed = end_time - start_time
    print(f"[K-Means-1] 完成一次聚类。总耗时: {elapsed:.2f} 秒，共处理 {len(results)} 张图片。")




# import os
# import cv2
# import json
# import numpy as np
# from sklearn.cluster import KMeans

# def extract_individual_colors(input_dir="datasets",
#                               output_json="output3/clustering_individual/colors.json",
#                               K=10):

#     os.makedirs(os.path.dirname(output_json), exist_ok=True)
#     results = {}

#     for img_name in os.listdir(input_dir):
#         if not img_name.lower().endswith((".jpg", ".png")):
#             continue

#         img = cv2.imread(os.path.join(input_dir, img_name))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # 仅使用非白背景的像素
#         mask = np.any(img_rgb < 240, axis=2)
#         pixels = img_rgb[mask].reshape(-1, 3)

#         km = KMeans(n_clusters=K, random_state=42).fit(pixels)
#         centers = km.cluster_centers_
#         labels = km.labels_

#         _, counts = np.unique(labels, return_counts=True)
#         ratios = (counts / counts.sum()).tolist()

#         results[img_name] = {
#             "centers": centers.tolist(),
#             "ratios": ratios
#         }

#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)

#     print("[K-Means-1] 完成一次聚类。")
