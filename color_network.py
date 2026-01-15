#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @File:color_network.py
# @Author:B站-夜游神人
# @Time:2026/01/12 23:48:07


import json
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from matplotlib.patches import Circle

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
# ==========================================================
# 安全 RGB 转换函数（防止 NaN/负数/超过255）
# ==========================================================
def safe_rgb(rgb):
    rgb = np.array(rgb, dtype=float)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    rgb = np.clip(rgb, 0, 255)
    rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


# ==========================================================
# 构建色彩网络图
# ==========================================================
def build_color_network(individual_json="output/clustering_individual/colors.json",
                        group_json="output/clustering_group/group_colors.json",
                        output_img="output/network/color_network.png"):

    os.makedirs(os.path.dirname(output_img), exist_ok=True)

    # ------------------------------------------------------
    # 加载 JSON 数据
    # ------------------------------------------------------
    with open(individual_json, "r", encoding="utf-8") as f:
        individuals = json.load(f)
    with open(group_json, "r", encoding="utf-8") as f:
        group = json.load(f)

    group_centers = np.array(group["group_centers"])

    # ------------------------------------------------------
    # 计算每个 group center 的实际占比（用于节点大小）
    # ------------------------------------------------------
    group_weights = np.zeros(len(group_centers))

    for _, info in individuals.items():
        centers = np.array(info["centers"])
        ratios = np.array(info["ratios"])

        for c, r in zip(centers, ratios):
            dist = np.linalg.norm(group_centers - c, axis=1)
            nearest = np.argmin(dist)
            group_weights[nearest] += r

    # 避免全 0 情况
    if group_weights.sum() > 0:
        group_weights = group_weights / group_weights.sum()
    else:
        group_weights = np.ones_like(group_weights) / len(group_weights)

    # ------------------------------------------------------
    # 按占比从大到小排序，并重新映射编号
    # ------------------------------------------------------
    sorted_idx = np.argsort(group_weights)[::-1]   # 最大 → 最小

    group_centers = group_centers[sorted_idx]
    group_weights = group_weights[sorted_idx]

    id_map = {old: new for new, old in enumerate(sorted_idx)}

    # ------------------------------------------------------
    # 构建网络（节点按排序后顺序添加）
    # ------------------------------------------------------
    G = nx.Graph()
    for new_id in range(len(group_centers)):

        rgb_raw = np.array(group_centers[new_id], dtype=float)
        rgb_raw = np.nan_to_num(rgb_raw, nan=0.0, posinf=255.0, neginf=0.0)
        rgb_raw = np.clip(rgb_raw, 0, 255)

        G.add_node(
            new_id,
            color=tuple(safe_rgb(rgb_raw)),
            rgb=rgb_raw.tolist(),
            weight=group_weights[new_id]
        )

    # ------------------------------------------------------
    # 统计共现边
    # ------------------------------------------------------
    for img_name, info in individuals.items():
        centers = np.array(info["centers"])
        ratios = np.array(info["ratios"])

        idx_list = []

        for c in centers:
            dist = np.linalg.norm(group_centers - c, axis=1)
            nearest = np.argmin(dist)
            idx_list.append(nearest)

        # 遍历组合
        for a, b in itertools.combinations(idx_list, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    # ------------------------------------------------------
    # 构建圆形布局（逆时针，从顶部开始）
    # ------------------------------------------------------
    n_nodes = len(G.nodes())
    radius = 5
    center = (0, 0)

    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    angles = angles - np.pi / 2  # 顶部为 0°

    pos = {}
    for i, angle in enumerate(angles):
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        pos[i] = (x, y)

    # ------------------------------------------------------
    # 绘图
    # ------------------------------------------------------
    plt.figure(figsize=(10, 10), dpi=150)
    ax = plt.gca()

    circle = Circle(center, radius, fill=False, color='gray', linestyle='--', alpha=0.3)
    ax.add_patch(circle)

    # 节点颜色
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]

    # 节点大小（归一化缩放）
    min_size = 800
    max_size = 3000
    w = np.array([G.nodes[n]["weight"] for n in G.nodes()])
    node_size = min_size + (w - w.min()) / (w.max() - w.min() + 1e-9) * (max_size - min_size)

    # 绘制节点（无边框）
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors='none',
        linewidths=0,
        alpha=0.95
    )

    # ------------------------------------------------------
    # 绘制共现边
    # ------------------------------------------------------
    if G.edges():
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(weights)
        min_w = min(weights)

        for (u, v), w in zip(G.edges(), weights):
            width = 1 + 8 * (w - min_w) / (max_w - min_w + 1e-9)
            #width = 2 + 8 * (w - min_w) / (max_w - min_w + 1e-9)
            alpha = 0.3 + 0.7 * (w - min_w) / (max_w - min_w + 1e-9)

            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=alpha,
                edge_color='black'
            )  #edge_color是共现边的颜色

    # ------------------------------------------------------
    # 绘制节点编号（排序后）
    # ------------------------------------------------------
    # node_labels = {i: f"{i+1:02d}" for i in G.nodes()}
    # nx.draw_networkx_labels(
    #     G, pos,
    #     labels=node_labels,
    #     font_size=14,
    #     font_weight='bold',
    #     font_color='black'
    # )

    # ------------------------------------------------------
    # 显示 RGB 图例（右侧）
    # ------------------------------------------------------
    legend_x = 1.0
    legend_y = 0.8

    for i in range(n_nodes):
        rgb = G.nodes[i]["rgb"]
        text = f"{i+1:02d}: RGB({rgb[0]:.0f}, {rgb[1]:.0f}, {rgb[2]:.0f})"

        plt.text(
            legend_x, legend_y - i * 0.04,
            text,
            fontsize=12,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
            
        )

    # ------------------------------------------------------
    # 图形设置
    # ------------------------------------------------------
    plt.axis("equal")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()

    # 控制台输出
    print(f"[Network] 色彩网络模型构建完成")
    print(f"[Network] 节点数: {G.number_of_nodes()}")
    print(f"[Network] 边数: {G.number_of_edges()}")

    return G, pos
