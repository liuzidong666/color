#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @File:visualize_colors_circle.py
# @Author:B站-夜游神人
# @Time:2026/01/12 23:48:33
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

import time
from tqdm import tqdm


def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))

def normalize_centers(centers):
    """
    将颜色中心标准化到 matplotlib 可接受的范围：
    1. 去除 NaN / inf
    2. 将 0-255 转为 0-1
    3. 截断非法值到 0-1 区间
    """
    centers = np.array(centers, dtype=float)
    
    # ① 去除 NaN / inf
    centers = np.nan_to_num(centers, nan=0.0, posinf=255.0, neginf=0.0)

    # ② 如果颜色值大于 1，说明是 0-255 范围，需要归一化
    if centers.max() > 1.0:
        centers = centers / 255.0

    # ③ 最终确保所有值都在 0-1
    centers = np.clip(centers, 0.0, 1.0)

    return centers


# =============================
# 设置全局字体为新罗马字体
# =============================
def set_times_new_roman_font():
    # 方法1: 尝试设置Times New Roman
    try:
        # 查找Times New Roman字体路径
        font_paths = [
            "C:/Windows/Fonts/times.ttf",  # Windows
            "C:/Windows/Fonts/timesbd.ttf",  # Windows Bold
            "C:/Windows/Fonts/timesi.ttf",  # Windows Italic
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",  # Linux
            "/System/Library/Fonts/Times New Roman.ttf",  # macOS
            "/Library/Fonts/Times New Roman.ttf",  # macOS
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                # 添加字体到matplotlib
                fm.fontManager.addfont(font_path)
                font_name = fm.FontProperties(fname=font_path).get_name()
                
                # 设置全局字体
                rcParams['font.family'] = 'serif'
                rcParams['font.serif'] = [font_name]
                rcParams['font.sans-serif'] = [font_name]  # 备用字体
                rcParams['axes.unicode_minus'] = False  # 正确显示负号
                print(f"[Font] 已设置字体为: {font_name}")
                return True
    
    except Exception as e:
        print(f"[Font] 设置新罗马字体失败: {e}")
    
    # 方法2: 使用matplotlib内置的serif字体
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    rcParams['axes.unicode_minus'] = False
    print("[Font] 使用默认serif字体（Times New Roman为优先选项）")
    return True


# =============================
# 一、可视化 —— 第一次聚类结果（单图）
# =============================

# ==========================================================
# 一、可视化 —— 第一次聚类结果（单图）进度条版
# ==========================================================
# ==========================================================
# 一、可视化 —— 第一次聚类结果（单图）进度条版（最终可覆盖版）
# ==========================================================
def visualize_individual_colors(
    individual_json="output/clustering_individual/colors.json",
    output_dir="output/visual/individual"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置新罗马字体
    set_times_new_roman_font()

    with open(individual_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_list = list(data.items())

    start_time = time.time()  # 开始计时
    print("\n开始进行一次聚类可视化！！！")

    # 使用 tqdm 显示进度条
    for img_name, info in tqdm(img_list, desc="Visualizing Images", ncols=100):
        centers = np.array(info["centers"])
        ratios = np.array(info["ratios"])

        # 按占比排序
        idx = np.argsort(ratios)[::-1]
        centers = centers[idx]
        ratios = ratios[idx]

        hex_colors = [rgb_to_hex(c) for c in centers]


        # ======================================================
        # ----- 1. 色带 Strip（按照占比 ratios 动态长度绘制）-----
        # ======================================================
        strip_h = 200
        total_w = 600  # strip 总宽度保持固定，用于相对比例

        # 创建背景 strip
        strip = np.zeros((strip_h, total_w, 3), dtype=np.uint8)

        # ---- 预计算每段的整数宽度（避免出现黑条） ----
        raw_w_list = [total_w * r for r in ratios]
        int_w_list = [max(int(w), 1) for w in raw_w_list]  # 每段至少 1 像素

        # 当前整数宽度总和
        current_sum = sum(int_w_list)
        diff = total_w - current_sum  # 正差值需要补；负差值需要减

        # ---- 将差值全部补偿给最大比例的颜色（确保总宽度一致）----
        if diff != 0:
            max_idx = np.argmax(ratios)
            int_w_list[max_idx] += diff

        # ---- 绘制 strip ----
        cur_x = 0
        for c, w in zip(centers, int_w_list):
            strip[:, cur_x:cur_x + w] = c
            cur_x += w

        plt.figure(figsize=(10, 2))
        plt.imshow(strip)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{img_name}_strip.png", dpi=300, bbox_inches='tight')
        plt.close()
        # ======================================================


        # ----- 2. 柱状图 -----
        plt.figure(figsize=(10, 5))
        safe_colors = normalize_centers(centers)
        bars = plt.bar(range(len(safe_colors)), ratios * 100, color=safe_colors)

        plt.xticks(range(len(centers)), hex_colors, rotation=45, ha="right", 
                   fontproperties=fm.FontProperties(family='serif'))
        plt.ylabel("Percentage %", fontproperties=fm.FontProperties(family='serif', size=12))
        plt.title("Color Percentage (Individual K-means)", 
                  fontproperties=fm.FontProperties(family='serif', size=14, weight='bold'))

        # 设置坐标轴标签字体
        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(fm.FontProperties(family='serif'))
        
        # 柱子上方的百分比数字
        for i, v in enumerate(ratios * 100):
            plt.text(i, v + 0.5, f"{v:.1f}%", 
                     ha="center", 
                     fontproperties=fm.FontProperties(family='serif', size=9),
                     fontweight='normal')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{img_name}_bar.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ----- 3. RGB/HEX 文本输出 -----
        with open(f"{output_dir}/{img_name}_color.txt", "w") as ff:
            for h, rgb, r in zip(hex_colors, centers, ratios):
                ff.write(f"{h}   {tuple(rgb.astype(int))}   {r*100:.2f}%\n")


    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[Visualize-1] 所有图片可视化完成，总耗时: {elapsed:.2f} 秒，共处理 {len(img_list)} 张图片。")



# =============================
# 二、可视化 —— 第二次聚类结果（群体）
# =============================
def visualize_group_colors(
    group_json="output/clustering_group/group_colors.json",
    individual_json="output/clustering_individual/colors.json",
    output_dir="output/visual/group"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置新罗马字体
    set_times_new_roman_font()

    with open(group_json, "r", encoding="utf-8") as f:
        group = json.load(f)

    with open(individual_json, "r", encoding="utf-8") as f:
        individuals = json.load(f)

    # 群体中心颜色
    group_centers = np.array(group["group_centers"])

    # -------------------------
    # 计算每个 group color 的全局比例（权重）
    # -------------------------
    weights = np.zeros(len(group_centers))

    for _, info in individuals.items():
        centers = np.array(info["centers"])   # shape(K,3)
        ratios = np.array(info["ratios"])

        # 每个 center 找到最近的 group center
        for c, r in zip(centers, ratios):
            dist = np.linalg.norm(group_centers - c, axis=1)
            idx = np.argmin(dist)
            weights[idx] += r

    # 归一化
    weights = weights / weights.sum()

    # 按权重排序
    idx = np.argsort(weights)[::-1]
    group_centers = group_centers[idx]
    weights = weights[idx]

    hex_colors = [rgb_to_hex(c) for c in group_centers]

    # ----- 1. 色带 Strip（按照比例 weights 动态长度绘制）-----
    strip_h = 200
    total_w = 600  # 固定总宽度

    # 创建空白 strip
    strip = np.zeros((strip_h, total_w, 3), dtype=np.uint8)

    # ---- 预计算每段的整数宽度（避免黑条/缺口） ----
    raw_w_list = [total_w * w for w in weights]
    int_w_list = [max(int(w), 1) for w in raw_w_list]  # 每段至少 1 像素

    # 当前整数宽度总和
    current_sum = sum(int_w_list)
    diff = total_w - current_sum  # 正差值需要补，负差值需要减

    # 将差值补偿给最大比例的颜色
    if diff != 0:
        max_idx = np.argmax(weights)
        int_w_list[max_idx] += diff

    # 绘制 strip
    cur_x = 0
    for c, w in zip(group_centers, int_w_list):
        strip[:, cur_x:cur_x + w] = c
        cur_x += w

    plt.figure(figsize=(10, 2))
    plt.imshow(strip)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/group_strip.png", dpi=300, bbox_inches='tight')
    plt.close()


    # ----- 2. 柱状图 -----
    plt.figure(figsize=(10, 5))
    # bars = plt.bar(range(len(group_centers)), weights * 100,
    #                color=[c/255 for c in group_centers])
    safe_colors = normalize_centers(group_centers)
    
    bars = plt.bar(range(len(safe_colors)), weights * 100, color=safe_colors)

    plt.xticks(range(len(group_centers)), hex_colors, rotation=45, ha="right",
               fontproperties=fm.FontProperties(family='serif'))
    plt.ylabel("Percentage %", fontproperties=fm.FontProperties(family='serif', size=12))
    plt.title("Color Percentage (Group K-means)", 
              fontproperties=fm.FontProperties(family='serif', size=14, weight='bold'))

    # 设置坐标轴标签字体
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fm.FontProperties(family='serif'))
    
    # 设置坐标轴刻度字体
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    
    # 柱子上方的百分比数字
    for i, v in enumerate(weights * 100):
        plt.text(i, v + 0.5, f"{v:.1f}%", 
                 ha="center", 
                 fontproperties=fm.FontProperties(family='serif', size=9),
                 fontweight='normal')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/group_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----- 3. 环形图（Donut Chart） -----
    plt.figure(figsize=(10, 8))
    
    # 方法1：不使用labels参数，手动添加文本标签
    wedges, texts = plt.pie(
        weights * 100,
        colors=normalize_centers(group_centers),
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5),  # width控制环形宽度
    )
    
    # 设置wedge的轮廓颜色
    for wedge in wedges:
        wedge.set_edgecolor('white')
        wedge.set_linewidth(1.5)
    
    # 创建图例
    legend_labels = [f"{hex_colors[i]}: {weights[i]*100:.1f}%" for i in range(len(hex_colors))]
    plt.legend(
        wedges, 
        legend_labels,
        title="Colors",
        loc="center left",
        bbox_to_anchor=(0.9, 0, 0.5, 1),
        prop=fm.FontProperties(family='serif', size=12)
    )
    
    # 添加中心空白区域的百分比文本
    plt.text(0, 0, '100%', 
             ha='center', va='center', 
             fontproperties=fm.FontProperties(family='serif', size=20, weight='bold'))
    
    plt.title("Group Color Distribution\n(Donut Chart)", 
              fontproperties=fm.FontProperties(family='serif', size=16, weight='bold'),
              pad=20)
    
    # 设置长宽比相等
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/group_donut.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----- 4. 可选的另一种环形图样式（直接在图表上显示百分比） -----
    plt.figure(figsize=(10, 8))
    
    # 创建带百分比标签的环形图
    wedges, texts, autotexts = plt.pie(
        weights * 100,
        colors=normalize_centers(group_centers),
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5),
        autopct='%1.1f%%',  # 自动添加百分比
        pctdistance=0.9,   # 百分比距离中心的距离
    )
    
    # 设置百分比文本样式
    for autotext in autotexts:
        autotext.set_fontproperties(fm.FontProperties(family='serif', size=9, weight='bold'))
        autotext.set_color('white')
    
    # 添加中心文本
    plt.text(0, 0, '100%', 
             ha='center', va='center', 
             fontproperties=fm.FontProperties(family='serif', size=20, weight='bold'),
             color='black')
    
    plt.title("Group Color Distribution\n(Donut Chart with Percentages)", 
              fontproperties=fm.FontProperties(family='serif', size=16, weight='bold'),
              pad=20)
    
    # 添加图例
    plt.legend(
        wedges, 
        hex_colors,
        title="Color Codes",
        loc="center left",
        bbox_to_anchor=(0.85, 0, 0.5, 1),
        prop=fm.FontProperties(family='serif', size=12)#原始size=9
    )
    
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/group_donut_with_pct.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ----- 5. RGB/HEX 文本输出 -----
    with open(f"{output_dir}/group_color.txt", "w") as ff:
        for h, rgb, w in zip(hex_colors, group_centers, weights):
            ff.write(f"{h}   {tuple(rgb.astype(int))}   {w*100:.2f}%\n")

    print("[Visualize-2] 完成二次聚类可视化（群体色彩）")


# =============================
# 统一的字体设置函数（供其他模块使用）
# =============================
def setup_matplotlib_fonts():
    """
    统一设置matplotlib字体，供所有绘图函数调用
    """
    # 设置新罗马字体
    success = set_times_new_roman_font()
    
    # 额外设置数学字体（如果绘图中有数学公式）
    try:
        rcParams['mathtext.fontset'] = 'stix'  # STIX字体类似Times New Roman
        rcParams['mathtext.rm'] = 'Times New Roman'
        rcParams['mathtext.it'] = 'Times New Roman:italic'
        rcParams['mathtext.bf'] = 'Times New Roman:bold'
        print("[Font] 已设置数学字体")
    except:
        pass
    
    return success


# =============================
# 测试函数
# =============================
def test_font_setting():
    """测试字体设置是否成功"""
    setup_matplotlib_fonts()
    
    # 创建测试图
    plt.figure(figsize=(8, 4))
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 15, 25, 30]
    
    plt.plot(x, y, 'o-', linewidth=2)
    plt.title('Test Plot with Times New Roman', fontsize=14, fontweight='bold')
    plt.xlabel('X Axis', fontsize=12) #设置图中代码字体大小
    plt.ylabel('Y Axis', fontsize=12)
    plt.text(3, 20, f'Text: {y[2]:.1f}', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_font_output.png', dpi=150)
    plt.close()
    print("[Test] 字体测试图已保存为 test_font_output.png")


# =============================
# 主函数
# =============================
if __name__ == "__main__":
    # 测试字体设置
    test_font_setting()
    
    # 运行可视化函数
    # visualize_individual_colors()
    # visualize_group_colors()