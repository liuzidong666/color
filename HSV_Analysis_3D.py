#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @File:HSV_Analysis_3D.py
# @Author:B站-夜游神人
# @Time:2026/01/15 11:42:51
#--------------------提示--------------
#运行代码的文件夹修改入口：单个HSV分析在503行；整体HSV分析596行，请跳转修改！！！


import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


# ==========================================================
# 安全 RGB
# ==========================================================
def safe_rgb(rgb):
    rgb = np.array(rgb, dtype=float)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255, neginf=0)
    rgb = np.clip(rgb, 0, 255)
    return rgb / 255.0


# ==========================================================
# 文献公式 RGB → HSV
# ==========================================================
def rgb_to_hsv_manual(r, g, b):

    C_max = max(r, g, b)
    C_min = min(r, g, b)
    delta = C_max - C_min

    V = C_max
    S = 0 if C_max == 0 else delta / C_max

    if delta == 0:
        H_deg = 0
    else:
        if C_max == r:
            H_deg = ((g - b) / delta) % 6
        elif C_max == g:
            H_deg = (b - r) / delta + 2
        else:
            H_deg = (r - g) / delta + 4
        H_deg *= 60

    if H_deg < 0:
        H_deg += 360

    return np.deg2rad(H_deg), S, V


# ==========================================================
# RGB转HEX颜色
# ==========================================================
def rgb_to_hex(rgb):
    rgb_int = [int(round(c)) for c in rgb]
    return '#{:02x}{:02x}{:02x}'.format(rgb_int[0], rgb_int[1], rgb_int[2])


# ==========================================================
# 绘制环形图
# ==========================================================
def plot_donut_chart(colors, ratios, image_name, save_path):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    # 
    ratios = np.array(ratios)
    colors = np.array(colors)
    
    sort_idx = np.argsort(ratios)[::-1]
    ratios_sorted = ratios[sort_idx]
    colors_sorted = colors[sort_idx]
    
    percentages = ratios_sorted * 100
    
    wedges, texts, autotexts = ax.pie(
        percentages,
        colors=colors_sorted,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5),
        autopct='%1.1f%%',
        pctdistance=0.85,
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    total_percentage = np.sum(percentages)

    
    # ax.set_title("Color Percentage Donut Chart", fontsize=18, fontweight='bold')
    # ax.title.set_position([0.5, 1.0])  # 更靠近图像
    # fig.subplots_adjust(top=0.95)      # 调整整个图像顶部空白
    # 使用 suptitle
    fig.suptitle(
        "Color Percentage Donut Chart",
        fontsize=18,
        fontweight='bold',
        y=0.9  # y越小标题越靠近图像，范围约 0~1
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=False, bbox_inches='tight')
    plt.close()


# ==========================================================
# 雷达图
# ==========================================================
def plot_radar(theta, radius, colors, weights, title, save_path):

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_rlim(0, 1.0)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0],
                  labels=["0.2", "0.4", "0.6", "0.8", "1.0"],
                  angle=135, fontsize=14)

    ax.set_thetagrids(
        np.arange(0, 360, 45),
        labels=[f"{d}°" for d in np.arange(0, 360, 45)],
        fontsize=14
    )

    w = np.array(weights)
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-9)
    bubble_sizes = 200 + w_norm * (1800 - 200)

    ax.scatter(theta, radius, c=colors, s=bubble_sizes, alpha=0.9, linewidth=0)

    ax.set_title(title, fontsize=16, pad=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=False)
    plt.close()

# ==========================================================
# 新增：H-S 3D histogram
# ==========================================================
def plot_hs_histogram_3d(H_list, S_list, ratios, save_path, title="H-S 3D histogram", background_alpha=1.0):

    H_deg = np.rad2deg(H_list)
    H_bin = np.floor(H_deg / 36).astype(int)
    H_bin = np.clip(H_bin, 0, 9)

    S_bin = np.floor(S_list * 10).astype(int)
    S_bin = np.clip(S_bin, 0, 9)

    # 打印调试信息
    # print(f"原始H值（度）: {H_deg}")
    # print(f"H bins: {H_bin}")
    # print(f"原始S值: {S_list}")
    # print(f"S bins: {S_bin}")
    # print(f"颜色数量: {len(H_list)}")
    # print(f"H-S组合: {list(zip(H_bin, S_bin))}")
    
    hist_grid = np.zeros((10, 10))
    for h, s, r in zip(H_bin, S_bin, ratios):
        hist_grid[h, s] += r * 100

    # 打印非零的格子
    non_zero_cells = np.argwhere(hist_grid > 0)
    #print(f"非零格子数量: {len(non_zero_cells)}")
    # for h, s in non_zero_cells:
    #     print(f"  格子({h},{s}): {hist_grid[h,s]:.2f}%")

    _x = np.arange(10)
    _y = np.arange(10)
    xx, yy = np.meshgrid(_x, _y)
    x, y = xx.ravel(), yy.ravel()
    z = np.zeros_like(x)
    dx = dy = 0.8
    dz = hist_grid.T.ravel()

    # 打印柱体数量
    non_zero_bars = np.sum(dz > 0)
    #print(f"非零柱体数量: {non_zero_bars}")

    colors = []
    for hh, ss in zip(x, y):
        Hh = (hh + 0.5) * 36 / 360.0
        Ss = (ss + 0.5) / 10.0
        Vv = 0.9
        rgb = matplotlib.colors.hsv_to_rgb([Hh, Ss, Vv])
        colors.append(rgb)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 去除网格显示
    ax.grid(False)
    
    # 设置背景透明度和颜色
    # H-S面（底部，即xy平面，对应zaxis.pane）：设置为白色，可调透明度
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, background_alpha))
    
    # 两个侧面（xz和yz平面）：保持默认浅灰色或设置为浅灰色，可调透明度
    ax.xaxis.pane.set_facecolor((0.85, 0.85, 0.85, 0.65))
    ax.yaxis.pane.set_facecolor((0.85, 0.85, 0.85, 0.65))
    
    # 设置所有面板的边缘线
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor((0.8, 0.8, 0.8, 0.7))

    ax.bar3d(x, y, z, dx, dy, dz,
             color=colors,
             shade=True,
             alpha=0.75,
             edgecolor='black',
             linewidth=0.8)

    # ==========================================================
    # 新增：只对10个聚类颜色对应的零柱体绘制固定高度透明柱体
    # ==========================================================
    zero_height = 1.0  # 固定高度
    for h_bin_c, s_bin_c, rgb_color in zip(H_bin, S_bin, colors):
        # 仅当该bin的hist_grid高度为0时绘制
        if hist_grid[h_bin_c, s_bin_c] == 0:
            ax.bar3d(
                h_bin_c, s_bin_c, 0,
                dx, dy, zero_height,
                color=rgb_color,
                alpha=0.75,
                edgecolor='black',
                linewidth=0.8,
                shade=True
            )
    # ==========================================================

    ax.set_xlabel("Hue / H", fontsize=14)
    ax.set_ylabel("Saturation / S", fontsize=14)
    ax.set_zlabel("Percentage / %", fontsize=14)
    #ax.set_title(title, fontsize=16 ,pad=4)

    max_z = dz.max()
    ax.set_zlim(0, max(20, max_z + 2))

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.zaxis.set_major_locator(plt.MultipleLocator(2))

    fig.suptitle(
        title,
        fontsize=18,
        fontweight='bold',
        y=0.9  # y越小标题越靠近图像，范围约 0~1
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=False)
    plt.close()


# ==========================================================
# 新增：H-V 3D histogram
# ==========================================================
def plot_hv_histogram_3d(H_list, V_list, ratios, save_path, title="H-V 3D histogram", background_alpha=1.0):

    H_deg = np.rad2deg(H_list)
    H_bin = np.floor(H_deg / 36).astype(int)
    H_bin = np.clip(H_bin, 0, 9)

    V_bin = np.floor(V_list * 10).astype(int)
    V_bin = np.clip(V_bin, 0, 9)

    hist_grid = np.zeros((10, 10))
    for h, v, r in zip(H_bin, V_bin, ratios):
        hist_grid[h, v] += r * 100

    _x = np.arange(10)
    _y = np.arange(10)
    xx, yy = np.meshgrid(_x, _y)
    x, y = xx.ravel(), yy.ravel()
    z = np.zeros_like(x)
    dx = dy = 0.8
    dz = hist_grid.T.ravel()

    colors = []
    for hh, vv in zip(x, y):
        Hh = (hh + 0.5) * 36 / 360.0
        Vv = (vv + 0.5) / 10.0
        Ss = 0.9  # 固定饱和度高亮显示
        rgb = matplotlib.colors.hsv_to_rgb([Hh, Ss, Vv])
        colors.append(rgb)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, background_alpha))
    ax.xaxis.pane.set_facecolor((0.85, 0.85, 0.85, 0.65))
    ax.yaxis.pane.set_facecolor((0.85, 0.85, 0.85, 0.65))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor((0.8, 0.8, 0.8, 0.7))

    ax.bar3d(x, y, z, dx, dy, dz,
             color=colors, shade=True, alpha=0.75,
             edgecolor='black', linewidth=0.8)

    # 固定高度透明柱体
    zero_height = 1.0
    for h_bin_c, v_bin_c, rgb_color in zip(H_bin, V_bin, colors):
        if hist_grid[h_bin_c, v_bin_c] == 0:
            ax.bar3d(h_bin_c, v_bin_c, 0,
                     dx, dy, zero_height,
                     color=rgb_color, alpha=0.18,
                     edgecolor='black', linewidth=0.8,
                     shade=True)

    ax.set_xlabel("Hue / H", fontsize=14)
    ax.set_ylabel("Value / V", fontsize=14)
    ax.set_zlabel("Percentage / %", fontsize=14)
    max_z = dz.max()
    ax.set_zlim(0, max(20, max_z + 2))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.zaxis.set_major_locator(plt.MultipleLocator(2))

    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=False)
    plt.close()


# ==========================================================
# 新增：S-V 3D histogram
# ==========================================================
def plot_sv_histogram_3d(S_list, V_list, ratios, save_path, title="S-V 3D histogram", background_alpha=1.0):

    S_bin = np.floor(S_list * 10).astype(int)
    S_bin = np.clip(S_bin, 0, 9)

    V_bin = np.floor(V_list * 10).astype(int)
    V_bin = np.clip(V_bin, 0, 9)

    hist_grid = np.zeros((10, 10))
    for s, v, r in zip(S_bin, V_bin, ratios):
        hist_grid[s, v] += r * 100

    _x = np.arange(10)
    _y = np.arange(10)
    xx, yy = np.meshgrid(_x, _y)
    x, y = xx.ravel(), yy.ravel()
    z = np.zeros_like(x)
    dx = dy = 0.8
    dz = hist_grid.T.ravel()

    colors = []
    for ss, vv in zip(x, y):
        Ss = (ss + 0.5) / 10.0
        Vv = (vv + 0.5) / 10.0
        Hh = 0.0  # 固定色相红色系
        rgb = matplotlib.colors.hsv_to_rgb([Hh, Ss, Vv])
        colors.append(rgb)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, background_alpha))
    ax.xaxis.pane.set_facecolor((0.85, 0.85, 0.85, 0.65))
    ax.yaxis.pane.set_facecolor((0.85, 0.85, 0.85, 0.65))
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor((0.8, 0.8, 0.8, 0.7))

    ax.bar3d(x, y, z, dx, dy, dz,
             color=colors, shade=True, alpha=0.75,
             edgecolor='black', linewidth=0.8)

    zero_height = 1.0
    for s_bin_c, v_bin_c, rgb_color in zip(S_bin, V_bin, colors):
        if hist_grid[s_bin_c, v_bin_c] == 0:
            ax.bar3d(s_bin_c, v_bin_c, 0,
                     dx, dy, zero_height,
                     color=rgb_color, alpha=0.18,
                     edgecolor='black', linewidth=0.8,
                     shade=True)

    ax.set_xlabel("Saturation / S", fontsize=14)
    ax.set_ylabel("Value / V", fontsize=14)
    ax.set_zlabel("Percentage / %", fontsize=14)
    max_z = dz.max()
    ax.set_zlim(0, max(20, max_z + 2))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.zaxis.set_major_locator(plt.MultipleLocator(2))

    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=False)
    plt.close()


# ==========================================================
# 单张图片分析 - 加入 3D histogram
# ==========================================================
def analyze_single_image(image_data, image_name, output_dir):

    img_output_dir = os.path.join(output_dir, image_name.replace('.png', ''))
    os.makedirs(img_output_dir, exist_ok=True)
    
    centers = np.array(image_data["centers"], dtype=float)
    ratios = np.array(image_data["ratios"], dtype=float)
    
    color_list = []
    for rgb in centers:
        r, g, b = safe_rgb(rgb)
        color_list.append((r, g, b))
    
    donut_path = os.path.join(img_output_dir, f"{image_name}_donut_chart.png")
    plot_donut_chart(color_list, ratios, image_name, donut_path)
    #print(f"[Donut] {image_name} donut chart saved: {donut_path}")
    
    H_list, S_list, V_list = [], [], []
    
    for rgb in centers:
        r, g, b = safe_rgb(rgb)
        H, S, V = rgb_to_hsv_manual(r, g, b)
        H_list.append(H)
        S_list.append(S)
        V_list.append(V)
    
    H_list = np.array(H_list)
    S_list = np.array(S_list)
    V_list = np.array(V_list)
    
    order = np.argsort(H_list)
    H_list_sorted = H_list[order]
    S_list_sorted = S_list[order]
    V_list_sorted = V_list[order]
    ratios_sorted = ratios[order]
    color_list_sorted = [color_list[i] for i in order]
    
    hs_path = os.path.join(img_output_dir, f"{image_name}_HS_radar.png")
    plot_radar(
        H_list_sorted, S_list_sorted, color_list_sorted, ratios_sorted,
        f"H-S Distribution Radar Chart",
        hs_path
    )
    #print(f"[HS Radar] {image_name} H-S radar chart saved: {hs_path}")
    
    hv_path = os.path.join(img_output_dir, f"{image_name}_HV_radar.png")
    plot_radar(
        H_list_sorted, V_list_sorted, color_list_sorted, ratios_sorted,
        f"H-V Distribution Radar Chart",
        hv_path
    )
    #print(f"[HV Radar] {image_name} H-V radar chart saved: {hv_path}")

    # 新增：H-S 3D histogram
    hs3d_path = os.path.join(img_output_dir, f"{image_name}_HS_3D_hist.png")
    plot_hs_histogram_3d(H_list_sorted, S_list_sorted, ratios_sorted, hs3d_path,
                         title=f"H-S 3D histogram")
    #print(f"[HS 3D] {image_name} HS 3D histogram saved: {hs3d_path}")

    # H-V 3D histogram
    hv3d_path = os.path.join(img_output_dir, f"{image_name}_HV_3D_hist.png")
    plot_hv_histogram_3d(H_list_sorted, V_list_sorted, ratios_sorted, hv3d_path,
                         title="H-V 3D histogram")
    #print(f"[HV 3D] {image_name} HV 3D histogram saved: {hv3d_path}")

    # S-V 3D histogram
    sv3d_path = os.path.join(img_output_dir, f"{image_name}_SV_3D_hist.png")
    plot_sv_histogram_3d(S_list_sorted, V_list_sorted, ratios_sorted, sv3d_path,
                         title="S-V 3D histogram")
    #print(f"[SV 3D] {image_name} SV 3D histogram saved: {sv3d_path}")


    return {
        "donut_chart": donut_path,
        "hs_radar": hs_path,
        "hv_radar": hv_path,
        "hs_3d_hist": hs3d_path,
        "hv_3d_hist":hv3d_path,
        "sv_3d_hist":sv3d_path
    }


# ==========================================================
# 主函数：单张图 HSV 分析           对应运行时选择  “2”
# ==========================================================
import os
import json
import csv
import time
import numpy as np
from tqdm import tqdm  # 进度条

def individual_hsv_analysis(
    json_path="output_pao/clustering_individual/colors.json",
    output_dir="output_pao/hsv_individual",
    csv_name="hsv_analysis_report.csv"
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: JSON format error {json_path}")
        return
    
    csv_path = os.path.join(output_dir, csv_name)
    fieldnames = [
        "Analyzing image", "Number of colors", "Total ratio",
        "H_values_deg", "H_bins", "S_values", "S_bins",
        "Number of colors", "Non-zero cells", "Non-zero bars"
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # tqdm 进度条遍历
        for image_name, image_data in tqdm(data.items(), desc="HSV Analyzing images"):
            
            centers = np.array(image_data["centers"], dtype=float)
            ratios = np.array(image_data["ratios"], dtype=float)
            
            H_list, S_list, V_list = [], [], []
            for rgb in centers:
                r, g, b = safe_rgb(rgb)
                H, S, V = rgb_to_hsv_manual(r, g, b)
                H_list.append(H)
                S_list.append(S)
                V_list.append(V)
            H_list = np.array(H_list)
            S_list = np.array(S_list)
            
            # H-S bins 和非零统计
            H_deg = np.rad2deg(H_list)
            H_bin = np.floor(H_deg / 36).astype(int)
            H_bin = np.clip(H_bin, 0, 9)
            S_bin = np.floor(S_list * 10).astype(int)
            S_bin = np.clip(S_bin, 0, 9)
            
            hist_grid = np.zeros((10, 10))
            for h, s, r in zip(H_bin, S_bin, ratios):
                hist_grid[h, s] += r * 100
            non_zero_cells = np.argwhere(hist_grid > 0)
            dz = hist_grid.T.ravel()
            non_zero_bars = np.sum(dz > 0)
            
            # 写入 CSV
            writer.writerow({
                "Analyzing image": image_name,
                "Number of colors": len(centers),
                "Total ratio": round(float(np.sum(ratios)), 4),
                "H_values_deg": ";".join([f"{h:.2f}" for h in H_deg]),
                "H_bins": ";".join(map(str, H_bin)),
                "S_values": ";".join([f"{s:.2f}" for s in S_list]),
                "S_bins": ";".join(map(str, S_bin)),
                "Number of colors": len(H_list),
                "Non-zero cells": len(non_zero_cells),
                "Non-zero bars": int(non_zero_bars)
            })
            
            # 绘图（保留 analyze_single_image 功能，不打印信息）
            analyze_single_image(image_data, image_name, output_dir)
    
    elapsed = time.time() - start_time
    print(f"\n[Complete] All images analyzed successfully!")
    print(f"CSV report saved: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Total elapsed time: {elapsed:.2f} seconds")


# ==========================================================
# 整体色彩 HSV 分析（修改版，保持单图分析风格）  对应运行时选择  “1”
# ==========================================================
import os
import re
import time
import numpy as np

def hsv_analysis(
    txt_path="output_pao/visual/group/group_color.txt",
    output_dir="output_pao/hsv_final_group"
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    HEX, RGB, RATIO = [], [], []

    rgb_pattern = re.compile(r"np\.int64\((\d+)\)|(\d+)")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split()
            hex_code = parts[0]

            numbers = [int(m[0] or m[1]) for m in rgb_pattern.findall(line)]
            rgb_vals = tuple(numbers[:3])

            ratio_str = parts[-1].replace("%", "")
            ratio_val = float(ratio_str) / 100.0

            HEX.append(hex_code)
            RGB.append(rgb_vals)
            RATIO.append(ratio_val)

    RGB = np.array(RGB, dtype=float)
    weights = np.array(RATIO)

    color_list, H_list, S_list, V_list = [], [], [], []

    for rgb in RGB:
        r, g, b = safe_rgb(rgb)
        H, S, V = rgb_to_hsv_manual(r, g, b)
        color_list.append((r, g, b))
        H_list.append(H)
        S_list.append(S)
        V_list.append(V)

    H_list = np.array(H_list)
    S_list = np.array(S_list)
    V_list = np.array(V_list)

    # 排序
    order = np.argsort(H_list)
    H_list = H_list[order]
    S_list = S_list[order]
    V_list = V_list[order]
    weights = weights[order]
    color_list = [color_list[i] for i in order]

    # =====================
    # 绘制环形图
    # =====================
    donut_path = os.path.join(output_dir, "group_donut_chart.png")
    plot_donut_chart(color_list, weights, "Group Colors", donut_path)

    # =====================
    # H-S / H-V 雷达图
    # =====================
    hs_path = os.path.join(output_dir, "group_HS_radar.png")
    plot_radar(H_list, S_list, color_list, weights, "H-S Color Distribution Radar Chart", hs_path)

    hv_path = os.path.join(output_dir, "group_HV_radar.png")
    plot_radar(H_list, V_list, color_list, weights, "H-V Color Distribution Radar Chart", hv_path)

    # =====================
    # H-S / H-V / S-V 3D histogram
    # =====================
    hs3d_path = os.path.join(output_dir, "group_HS_3D_hist.png")
    plot_hs_histogram_3d(H_list, S_list, weights, hs3d_path, title="H-S 3D histogram")

    hv3d_path = os.path.join(output_dir, "group_HV_3D_hist.png")
    plot_hv_histogram_3d(H_list, V_list, weights, hv3d_path, title="H-V 3D histogram")

    sv3d_path = os.path.join(output_dir, "group_SV_3D_hist.png")
    plot_sv_histogram_3d(S_list, V_list, weights, sv3d_path, title="S-V 3D histogram")

    elapsed = time.time() - start_time
    print(f"\n[Complete] Group HSV analysis finished!")
    print(f"Output directory: {output_dir}")
    print(f"Total elapsed time: {elapsed:.2f} seconds")

# ==========================================================
# 主入口
# ==========================================================
if __name__ == "__main__":
    print("Analysis Mode Selection:")
    print("1. Group analysis (using original txt file)")
    print("2. Individual image analysis (using colors.json file)")
    
    choice = input("Please enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        hsv_analysis()
    elif choice == "2":
        individual_hsv_analysis()
    else:
        print("Invalid choice, default to individual image analysis")
        individual_hsv_analysis()
