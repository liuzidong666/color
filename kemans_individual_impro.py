import os
import cv2
import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm
import time


def kmeans_ultra_extract(
        input_dir="datasets",
        output_json="output3/clustering_individual/colors.json",
        K=10,
        sample_max=150000,
        bg_thresh_v=0.90,
        bg_thresh_s=0.20,
        spatial_weight=0.15,
        min_ratio=0.01
    ):
    """
    Ultra 版 KMeans 色彩提取（终极 KMeans++ 改进版）
    完全基于 KMeans，不使用任何其它聚类模型
    """

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    results = {}

    img_list = [f for f in os.listdir(input_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    start = time.time()

    for img_name in tqdm(img_list, desc="KMeans++ Ultra", ncols=100):

        path = os.path.join(input_dir, img_name)
        img = cv2.imread(path)
        if img is None:
            continue

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

        # ===============================
        # 1) 背景剔除（高亮 + 低饱和）
        # ===============================
        V = hsv[..., 2]
        S = hsv[..., 1]
        mask = ~((V > bg_thresh_v) & (S < bg_thresh_s))

        lab_pixels = lab[mask]
        rgb_pixels = rgb[mask]
        coords = np.indices((h, w)).transpose(1, 2, 0)[mask]

        if len(lab_pixels) == 0:
            continue

        # ===============================
        # 2) 随机采样
        # ===============================
        if len(lab_pixels) > sample_max:
            idx = np.random.choice(len(lab_pixels), sample_max, replace=False)
            lab_pixels = lab_pixels[idx]
            rgb_pixels = rgb_pixels[idx]
            coords = coords[idx]

        # ===============================
        # 3) 构建特征（LAB + 空间坐标）
        # ===============================
        coords = coords.astype(np.float32)
        coords /= max(h, w)
        features = np.concatenate([
            lab_pixels,                         # 颜色特征
            spatial_weight * coords             # 空间特征
        ], axis=1)

        # ===============================
        # 4) 使用 MiniBatchKMeans + KMeans++ 初始化
        # ===============================
        kmeans = MiniBatchKMeans(
            n_clusters=K,
            init="k-means++",
            random_state=42,
            n_init=10,
            batch_size=4096,
            max_iter=50,
            reassignment_ratio=0.01
        ).fit(features)

        labels = kmeans.labels_
        centers_lab = kmeans.cluster_centers_[:, :3]

        # ===============================
        # 5) 统计颜色比例
        # ===============================
        uniq, counts = np.unique(labels, return_counts=True)
        ratios = counts / counts.sum()

        centers_rgb = []
        for u in uniq:
            centers_rgb.append(rgb_pixels[labels == u].mean(axis=0))
        centers_rgb = np.array(centers_rgb)

        # ===============================
        # 6) 小簇合并（<1%）
        # ===============================
        merge_idx = []
        for i, r in enumerate(ratios):
            if r < min_ratio:
                merge_idx.append(i)

        if merge_idx:
            keep = [i for i in range(len(ratios)) if i not in merge_idx]
            kept_centers = centers_rgb[keep]
            kept_ratios = ratios[keep]

            for i in merge_idx:
                d = np.linalg.norm(centers_rgb[i] - kept_centers, axis=1)
                j = np.argmin(d)
                total = kept_ratios[j] + ratios[i]
                kept_centers[j] = (
                    kept_centers[j] * kept_ratios[j] +
                    centers_rgb[i] * ratios[i]
                ) / total
                kept_ratios[j] = total

            centers_rgb = kept_centers
            ratios = kept_ratios

        # ===============================
        # 7) Gamma 校正（视觉增强）
        # ===============================
        gamma = 1 / 2.2
        centers_rgb = np.power(centers_rgb / 255.0, gamma) * 255.0
        centers_rgb = centers_rgb.clip(0, 255)

        # ===============================
        # 8) 按比例排序输出
        # ===============================
        idx = np.argsort(-ratios)
        centers_rgb = centers_rgb[idx]
        ratios = ratios[idx]

        results[img_name] = {
            "centers": centers_rgb.tolist(),
            "ratios": ratios.tolist()
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[KMeans++ Ultra] 完成。耗时 {time.time() - start:.2f} 秒，处理 {len(results)} 张图片。")









# import os
# import cv2
# import json
# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# from tqdm import tqdm
# import time

# def extract_individual_colors(
#         input_dir="datasets",
#         output_json="output3/clustering_individual/colors.json",
#         K=10,
#         sample_max=200000,              # 最大采样像素数（过大会变慢）
#         hsv_bg_remove=True,             # 使用 HSV 背景剔除
#         sat_thresh=0.25,                # 饱和度阈值（背景 S<0.25）
#         val_thresh=0.92,                # 明度阈值（背景 V>0.92）
#         min_ratio=0.005                 # 最小比例过滤（小簇合并）
#     ):

#     os.makedirs(os.path.dirname(output_json), exist_ok=True)
#     results = {}

#     img_list = [f for f in os.listdir(input_dir)
#                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

#     start_time = time.time()

#     for img_name in tqdm(img_list, desc="Enhanced Color Clustering", ncols=100):
#         path = os.path.join(input_dir, img_name)
#         img = cv2.imread(path)
#         if img is None:
#             continue
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # ==========================================================
#         # A. HSV 背景剔除（增强精度）
#         # ==========================================================
#         if hsv_bg_remove:
#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
#             H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]

#             mask = ~((S < sat_thresh) & (V > val_thresh))   # 不要低饱和 + 高亮度
#         else:
#             mask = np.any(rgb < 240, axis=2)

#         pixels = rgb[mask].reshape(-1, 3)
#         if len(pixels) == 0:
#             continue

#         # ==========================================================
#         # C. 像素随机抽样（提升稳定性）
#         # ==========================================================
#         if pixels.shape[0] > sample_max:
#             idx = np.random.choice(pixels.shape[0], sample_max, replace=False)
#             pixels = pixels[idx]

#         # ==========================================================
#         # B. 使用 MiniBatchKMeans（更稳、更快）
#         # ==========================================================
#         km = MiniBatchKMeans(
#             n_clusters=K,
#             random_state=42,
#             batch_size=4096,
#             max_iter=50,
#             reassignment_ratio=0.01
#         ).fit(pixels)

#         centers = km.cluster_centers_
#         labels = km.labels_

#         # ==========================================================
#         # D. 过滤比例非常小的簇，防止噪声影响颜色
#         # ==========================================================
#         uniq, counts = np.unique(labels, return_counts=True)
#         ratios = counts / counts.sum()

#         # 小于 min_ratio 的簇合并到最近的大簇
#         large_idx = np.where(ratios >= min_ratio)[0]
#         small_idx = np.where(ratios < min_ratio)[0]

#         if len(small_idx) > 0 and len(large_idx) > 0:
#             for si in small_idx:
#                 # 找最近的大簇
#                 d = np.linalg.norm(centers[si] - centers[large_idx], axis=1)
#                 nearest = large_idx[np.argmin(d)]
#                 # 合并：重新分配比例 + 颜色中心平滑
#                 total = counts[si] + counts[nearest]
#                 centers[nearest] = (
#                     centers[nearest] * counts[nearest] +
#                     centers[si] * counts[si]
#                 ) / total
#                 counts[nearest] += counts[si]
#                 counts[si] = 0

#             # 去掉空簇
#             mask_nonzero = counts > 0
#             centers = centers[mask_nonzero]
#             ratios = (counts[mask_nonzero] / counts[mask_nonzero].sum()).tolist()
#         else:
#             ratios = ratios.tolist()

#         # ==========================================================
#         # E. 颜色感知校正（Gamma 校正）
#         # ==========================================================
#         gamma = 1.0 / 2.2
#         centers_corrected = np.power(centers / 255.0, gamma) * 255.0
#         centers_corrected = np.clip(centers_corrected, 0, 255)

#         results[img_name] = {
#             "centers": centers_corrected.tolist(),
#             "ratios": ratios
#         }

#     # 保存结果
#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)

#     print(f"[Enhanced K-Means] 完成。总耗时 {time.time() - start_time:.2f} 秒，处理 {len(results)} 张图片。")
