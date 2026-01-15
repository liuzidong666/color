import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json

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
    """将RGB值转换为HEX颜色码"""
    rgb_int = [int(round(c)) for c in rgb]
    return '#{:02x}{:02x}{:02x}'.format(rgb_int[0], rgb_int[1], rgb_int[2])


# ==========================================================
# 绘制环形图（修改版 - 在扇形中心显示百分比）
# ==========================================================
def plot_donut_chart(colors, ratios, image_name, save_path):
    """
    绘制环形饼图（Donut Chart）
    colors: 颜色列表，每个颜色为(r,g,b)格式，值在[0,1]
    ratios: 比例列表
    image_name: 图片名称
    save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # 计算百分比并排序（从大到小）
    ratios = np.array(ratios)
    colors = np.array(colors)
    
    # 按比例从大到小排序（顺时针绘制）
    sort_idx = np.argsort(ratios)[::-1]  # 从大到小
    ratios_sorted = ratios[sort_idx]
    colors_sorted = colors[sort_idx]
    
    # 计算百分比
    percentages = ratios_sorted * 100
    
    # 将RGB颜色转换为HEX颜色码用于显示
    hex_colors = []
    for color in colors_sorted * 255:  # 转换回0-255范围
        hex_colors.append(rgb_to_hex(color))
    
    # 绘制环形图 - 顺时针从大到小，自动添加百分比
    wedges, texts, autotexts = ax.pie(
        percentages,  # 使用百分比值
        colors=colors_sorted,
        startangle=90,
        counterclock=False,  # 顺时针方向
        wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5),
        autopct='%1.1f%%',   # 自动添加百分比
        pctdistance=0.85,    # 百分比距离中心的距离
    )
    
    # 设置百分比文本样式 - 白色粗体
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # 添加中心文本   可以去除；直接注释即可
    total_percentage = np.sum(percentages)
    ax.text(0, 0, f'{total_percentage:.1f}%', 
             ha='center', va='center', 
             fontsize=20, fontweight='bold',
             color='black')
    
    # 设置标题
    ax.set_title(f"Color Percentage Donut Chart\n{image_name}", 
                 fontsize=16, fontweight='bold',
                 pad=10)
    
    # 设置长宽比相等
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
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
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8,1.0],
                  labels=["0.2", "0.4", "0.6", "0.8","1.0"],
                  angle=135, fontsize=14)#调整图片信息标注位置和大小

    ax.set_thetagrids(
        np.arange(0, 360, 45),
        labels=[f"{d}°" for d in np.arange(0, 360, 45)],
        fontsize=14
    )

    # 节点大小按占比
    w = np.array(weights)
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-9)
    bubble_sizes = 200 + w_norm * (1800 - 200)

    ax.scatter(theta, radius, c=colors, s=bubble_sizes, alpha=0.9, linewidth=0)

    ax.set_title(title, fontsize=18, pad=25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.close()


# ==========================================================
# 单张图片HSV分析（新增函数）
# ==========================================================
def analyze_single_image(image_data, image_name, output_dir):
    """
    分析单张图片的颜色数据
    image_data: 包含centers和ratios的字典
    image_name: 图片文件名
    output_dir: 输出目录
    """
    # 创建图片专属的输出目录
    img_output_dir = os.path.join(output_dir, image_name.replace('.png', ''))
    os.makedirs(img_output_dir, exist_ok=True)
    
    # 提取颜色中心和比例
    centers = np.array(image_data["centers"], dtype=float)
    ratios = np.array(image_data["ratios"], dtype=float)
    
    # 转换为安全的RGB颜色
    color_list = []
    for rgb in centers:
        r, g, b = safe_rgb(rgb)
        color_list.append((r, g, b))
    
    # 1. 绘制环形图
    donut_path = os.path.join(img_output_dir, f"{image_name}_donut_chart.png")
    plot_donut_chart(color_list, ratios, image_name, donut_path)
    print(f"[Donut] {image_name} donut chart saved: {donut_path}")
    
    # 2. 计算HSV并绘制雷达图
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
    
    # 按色相排序
    order = np.argsort(H_list)
    H_list_sorted = H_list[order]
    S_list_sorted = S_list[order]
    V_list_sorted = V_list[order]
    ratios_sorted = ratios[order]
    color_list_sorted = [color_list[i] for i in order]
    
    # H-S 雷达图
    hs_path = os.path.join(img_output_dir, f"{image_name}_HS_radar.png")
    plot_radar(
        H_list_sorted, S_list_sorted, color_list_sorted, ratios_sorted,
        f"H-S Distribution Radar Chart - {image_name}",
        hs_path
    )
    print(f"[HS Radar] {image_name} H-S radar chart saved: {hs_path}")
    
    # H-V 雷达图
    hv_path = os.path.join(img_output_dir, f"{image_name}_HV_radar.png")
    plot_radar(
        H_list_sorted, V_list_sorted, color_list_sorted, ratios_sorted,
        f"H-V Distribution Radar Chart - {image_name}",
        hv_path
    )
    print(f"[HV Radar] {image_name} H-V radar chart saved: {hv_path}")
    
    return {
        "donut_chart": donut_path,
        "hs_radar": hs_path,
        "hv_radar": hv_path
    }


# ==========================================================
# 主函数：处理单张图片分析
# ==========================================================
def individual_hsv_analysis(
    json_path="output/clustering_individual/colors.json",
    output_dir="output2/hsv_individual"
):
    """
    处理单张图片的HSV分析
    json_path: 包含各图片颜色聚类结果的JSON文件路径
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: JSON format error {json_path}")
        return
    
    # 分析每张图片
    results = {}
    for image_name, image_data in data.items():
        print(f"\nAnalyzing image: {image_name}")
        print(f"Number of colors: {len(image_data['centers'])}")
        print(f"Total ratio: {sum(image_data['ratios']):.4f}")
        
        result_paths = analyze_single_image(image_data, image_name, output_dir)
        results[image_name] = result_paths
    
    # 生成分析报告
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Individual Image HSV Analysis Report\n")
        f.write("=" * 50 + "\n")
        for image_name, paths in results.items():
            f.write(f"\nImage: {image_name}\n")
            f.write(f"  Donut Chart: {paths['donut_chart']}\n")
            f.write(f"  H-S Radar Chart: {paths['hs_radar']}\n")
            f.write(f"  H-V Radar Chart: {paths['hv_radar']}\n")
    
    print(f"\n[Complete] All images analyzed successfully!")
    print(f"Report saved: {report_path}")
    print(f"Output directory: {output_dir}")


# ==========================================================
# 原有的整体分析函数（保持不变）
# ==========================================================
def hsv_analysis(
    txt_path="output/visual/group/group_color.txt",
    output_dir="output1/hsv_final"
):
    """原有的整体分析函数，保持不变"""
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

    H_list, S_list, V_list, color_list = [], [], [], []

    for rgb in RGB:
        r, g, b = safe_rgb(rgb)
        H, S, V = rgb_to_hsv_manual(r, g, b)

        H_list.append(H)
        S_list.append(S)
        V_list.append(V)
        color_list.append((r, g, b))

    H_list = np.array(H_list)
    S_list = np.array(S_list)
    V_list = np.array(V_list)

    order = np.argsort(H_list)
    H_list = H_list[order]
    S_list = S_list[order]
    V_list = V_list[order]
    weights = weights[order]
    color_list = [color_list[i] for i in order]

    plot_radar(
        H_list, S_list, color_list, weights,
        "H-S Color Distribution Radar Chart",
        f"{output_dir}/HS_radar.png"
    )

    plot_radar(
        H_list, V_list, color_list, weights,
        "H-V Color Distribution Radar Chart",
        f"{output_dir}/HV_radar.png"
    )

    print("[HSV] Group HSV analysis completed. Output directory:", output_dir)


# ==========================================================
# 主程序入口
# ==========================================================
if __name__ == "__main__":
    # 可以根据需要选择运行整体分析或单张图片分析
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



# import numpy as np
# import matplotlib.pyplot as plt
# import re
# import os
# import json
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["axes.unicode_minus"] = False


# # ==========================================================
# # 安全 RGB
# # ==========================================================
# def safe_rgb(rgb):
#     rgb = np.array(rgb, dtype=float)
#     rgb = np.nan_to_num(rgb, nan=0.0, posinf=255, neginf=0)
#     rgb = np.clip(rgb, 0, 255)
#     return rgb / 255.0


# # ==========================================================
# # 文献公式 RGB → HSV
# # ==========================================================
# def rgb_to_hsv_manual(r, g, b):

#     C_max = max(r, g, b)
#     C_min = min(r, g, b)
#     delta = C_max - C_min

#     V = C_max
#     S = 0 if C_max == 0 else delta / C_max

#     if delta == 0:
#         H_deg = 0
#     else:
#         if C_max == r:
#             H_deg = ((g - b) / delta) % 6
#         elif C_max == g:
#             H_deg = (b - r) / delta + 2
#         else:
#             H_deg = (r - g) / delta + 4
#         H_deg *= 60

#     if H_deg < 0:
#         H_deg += 360

#     return np.deg2rad(H_deg), S, V


# # ==========================================================
# # RGB转HEX颜色
# # ==========================================================
# def rgb_to_hex(rgb):
#     """将RGB值转换为HEX颜色码"""
#     rgb_int = [int(round(c)) for c in rgb]
#     return '#{:02x}{:02x}{:02x}'.format(rgb_int[0], rgb_int[1], rgb_int[2])


# # ==========================================================
# # 绘制环形图（修改版 - 在扇形中心显示百分比）
# # ==========================================================
# def plot_donut_chart(colors, ratios, image_name, save_path):
#     """
#     绘制环形饼图（Donut Chart）
#     colors: 颜色列表，每个颜色为(r,g,b)格式，值在[0,1]
#     ratios: 比例列表
#     image_name: 图片名称
#     save_path: 保存路径
#     """
#     fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
#     # 计算百分比并排序（从大到小）
#     ratios = np.array(ratios)
#     colors = np.array(colors)
    
#     # 按比例从大到小排序（顺时针绘制）
#     sort_idx = np.argsort(ratios)[::-1]  # 从大到小
#     ratios_sorted = ratios[sort_idx]
#     colors_sorted = colors[sort_idx]
    
#     # 计算百分比
#     percentages = ratios_sorted * 100
    
#     # 将RGB颜色转换为HEX颜色码用于显示
#     hex_colors = []
#     for color in colors_sorted * 255:  # 转换回0-255范围
#         hex_colors.append(rgb_to_hex(color))
    
#     # 绘制环形图 - 顺时针从大到小，自动添加百分比
#     wedges, texts, autotexts = ax.pie(
#         percentages,  # 使用百分比值
#         colors=colors_sorted,
#         startangle=90,
#         counterclock=False,  # 顺时针方向
#         wedgeprops=dict(width=0.3, edgecolor='white', linewidth=1.5),
#         autopct='%1.1f%%',   # 自动添加百分比
#         pctdistance=0.85,    # 百分比距离中心的距离
#     )
    
#     # 设置百分比文本样式 - 白色粗体
#     for autotext in autotexts:
#         autotext.set_color('white')
#         autotext.set_fontsize(10)
#         autotext.set_fontweight('bold')
    
#     # 添加中心文本
#     total_percentage = np.sum(percentages)
#     ax.text(0, 0, f'{total_percentage:.1f}%', 
#              ha='center', va='center', 
#              fontsize=20, fontweight='bold',
#              color='black')
    
#     # 设置标题
#     ax.set_title(f"Color Percentage Donut Chart\n{image_name}", 
#                  fontsize=16, fontweight='bold',
#                  pad=20)
    
#     # 设置长宽比相等
#     ax.set_aspect('equal')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
#     plt.close()


# # ==========================================================
# # 雷达图
# # ==========================================================
# def plot_radar(theta, radius, colors, weights, title, save_path):

#     fig = plt.figure(figsize=(8, 8), dpi=150)
#     ax = fig.add_subplot(111, polar=True)

#     ax.set_theta_zero_location("N")
#     ax.set_theta_direction(-1)

#     ax.set_rlim(0, 1.0)
#     ax.set_rgrids([0.2, 0.4, 0.6, 0.8],
#                   labels=["0.2", "0.4", "0.6", "0.8"],
#                   angle=90, fontsize=11)

#     ax.set_thetagrids(
#         np.arange(0, 360, 45),
#         labels=[f"{d}°" for d in np.arange(0, 360, 45)],
#         fontsize=12
#     )

#     # 节点大小按占比
#     w = np.array(weights)
#     w_norm = (w - w.min()) / (w.max() - w.min() + 1e-9)
#     bubble_sizes = 200 + w_norm * (1800 - 200)

#     ax.scatter(theta, radius, c=colors, s=bubble_sizes, alpha=0.9, linewidth=0)

#     ax.set_title(title, fontsize=18, pad=25)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, transparent=True)
#     plt.close()


# # ==========================================================
# # H-S 3D直方图
# # ==========================================================
# def plot_hs_3d_histogram(H_list, S_list, weights, image_name, save_path):
#     """
#     绘制H-S 3D直方图
#     H_list: 色相值列表（弧度）
#     S_list: 饱和度列表
#     weights: 权重（颜色占比）
#     image_name: 图片名称
#     save_path: 保存路径
#     """
#     fig = plt.figure(figsize=(12, 8), dpi=150)
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 将色相从弧度转换为角度
#     H_deg = np.degrees(H_list)
    
#     # 定义H和S的bins
#     h_bins = 12  # 12个色相区间（每个区间30度）
#     s_bins = 10  # 10个饱和度区间
    
#     # 创建H-S网格
#     h_edges = np.linspace(0, 360, h_bins + 1)
#     s_edges = np.linspace(0, 1, s_bins + 1)
    
#     # 计算2D直方图（H-S）
#     hist, h_edges, s_edges = np.histogram2d(
#         H_deg, S_list, 
#         bins=[h_bins, s_bins],
#         range=[[0, 360], [0, 1]],
#         weights=weights  # 使用权重
#     )
    
#     # 创建网格坐标
#     h_centers = (h_edges[:-1] + h_edges[1:]) / 2
#     s_centers = (s_edges[:-1] + s_edges[1:]) / 2
#     h_mesh, s_mesh = np.meshgrid(h_centers, s_centers)
    
#     # 转置hist以适应meshgrid
#     hist = hist.T
    
#     # 创建颜色映射 - 使用HSV颜色空间
#     # 将H值归一化到[0,1]用于颜色映射
#     h_norm = h_mesh / 360.0
#     colors = cm.hsv(h_norm)
    
#     # 调整颜色饱和度
#     for i in range(colors.shape[0]):
#         for j in range(colors.shape[1]):
#             # 保持色相，使用柱子的高度调整亮度
#             colors[i, j, 3] = 0.7 + 0.3 * (hist[i, j] / hist.max() if hist.max() > 0 else 0)
    
#     # 绘制3D柱状图
#     dx = 360 / h_bins * 0.8  # 柱子宽度
#     dy = 1.0 / s_bins * 0.8  # 柱子深度
    
#     # 使用bar3d绘制每个柱子
#     for i in range(h_bins):
#         for j in range(s_bins):
#             if hist[j, i] > 0:  # 只绘制有数据的柱子
#                 ax.bar3d(
#                     h_edges[i],  # x位置
#                     s_edges[j],  # y位置
#                     0,           # z起点
#                     dx,          # x宽度
#                     dy,          # y深度
#                     hist[j, i],  # z高度
#                     color=colors[j, i],
#                     alpha=0.8,
#                     edgecolor='k',
#                     linewidth=0.5
#                 )
    
#     # 设置坐标轴标签
#     ax.set_xlabel('Hue (degrees)', fontsize=12, labelpad=10)
#     ax.set_ylabel('Saturation', fontsize=12, labelpad=10)
#     ax.set_zlabel('Count (Weighted)', fontsize=12, labelpad=10)
    
#     # 设置坐标轴范围
#     ax.set_xlim(0, 360)
#     ax.set_ylim(0, 1)
    
#     # 设置视角
#     ax.view_init(elev=30, azim=45)
    
#     # 设置标题
#     ax.set_title(f"H-S 3D Histogram\n{image_name}", 
#                  fontsize=16, fontweight='bold',
#                  pad=20)
    
#     # 添加网格
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()


# # ==========================================================
# # 单张图片HSV分析（新增函数）
# # ==========================================================
# def analyze_single_image(image_data, image_name, output_dir):
#     """
#     分析单张图片的颜色数据
#     image_data: 包含centers和ratios的字典
#     image_name: 图片文件名
#     output_dir: 输出目录
#     """
#     # 创建图片专属的输出目录
#     img_output_dir = os.path.join(output_dir, image_name.replace('.png', ''))
#     os.makedirs(img_output_dir, exist_ok=True)
    
#     # 提取颜色中心和比例
#     centers = np.array(image_data["centers"], dtype=float)
#     ratios = np.array(image_data["ratios"], dtype=float)
    
#     # 转换为安全的RGB颜色
#     color_list = []
#     for rgb in centers:
#         r, g, b = safe_rgb(rgb)
#         color_list.append((r, g, b))
    
#     # 1. 绘制环形图
#     donut_path = os.path.join(img_output_dir, f"{image_name}_donut_chart.png")
#     plot_donut_chart(color_list, ratios, image_name, donut_path)
#     print(f"[Donut] {image_name} donut chart saved: {donut_path}")
    
#     # 2. 计算HSV并绘制雷达图
#     H_list, S_list, V_list = [], [], []
    
#     for rgb in centers:
#         r, g, b = safe_rgb(rgb)
#         H, S, V = rgb_to_hsv_manual(r, g, b)
#         H_list.append(H)
#         S_list.append(S)
#         V_list.append(V)
    
#     H_list = np.array(H_list)
#     S_list = np.array(S_list)
#     V_list = np.array(V_list)
    
#     # 按色相排序
#     order = np.argsort(H_list)
#     H_list_sorted = H_list[order]
#     S_list_sorted = S_list[order]
#     V_list_sorted = V_list[order]
#     ratios_sorted = ratios[order]
#     color_list_sorted = [color_list[i] for i in order]
    
#     # H-S 雷达图
#     hs_path = os.path.join(img_output_dir, f"{image_name}_HS_radar.png")
#     plot_radar(
#         H_list_sorted, S_list_sorted, color_list_sorted, ratios_sorted,
#         f"H-S Distribution Radar Chart - {image_name}",
#         hs_path
#     )
#     print(f"[HS Radar] {image_name} H-S radar chart saved: {hs_path}")
    
#     # H-V 雷达图
#     hv_path = os.path.join(img_output_dir, f"{image_name}_HV_radar.png")
#     plot_radar(
#         H_list_sorted, V_list_sorted, color_list_sorted, ratios_sorted,
#         f"H-V Distribution Radar Chart - {image_name}",
#         hv_path
#     )
#     print(f"[HV Radar] {image_name} H-V radar chart saved: {hv_path}")
    
#     # 3. H-S 3D直方图
#     hs_3d_path = os.path.join(img_output_dir, f"{image_name}_HS_3D_histogram.png")
#     plot_hs_3d_histogram(
#         H_list_sorted, S_list_sorted, ratios_sorted, 
#         image_name, hs_3d_path
#     )
#     print(f"[HS 3D] {image_name} H-S 3D histogram saved: {hs_3d_path}")
    
#     return {
#         "donut_chart": donut_path,
#         "hs_radar": hs_path,
#         "hv_radar": hv_path,
#         "hs_3d_histogram": hs_3d_path
#     }


# # ==========================================================
# # 主函数：处理单张图片分析
# # ==========================================================
# def individual_hsv_analysis(
#     json_path="output/clustering_individual/colors.json",
#     output_dir="output1/hsv_3d_individual"
# ):
#     """
#     处理单张图片的HSV分析
#     json_path: 包含各图片颜色聚类结果的JSON文件路径
#     output_dir: 输出目录
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 读取JSON文件
#     try:
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: File not found {json_path}")
#         return
#     except json.JSONDecodeError:
#         print(f"Error: JSON format error {json_path}")
#         return
    
#     # 分析每张图片
#     results = {}
#     for image_name, image_data in data.items():
#         print(f"\nAnalyzing image: {image_name}")
#         print(f"Number of colors: {len(image_data['centers'])}")
#         print(f"Total ratio: {sum(image_data['ratios']):.4f}")
        
#         result_paths = analyze_single_image(image_data, image_name, output_dir)
#         results[image_name] = result_paths
    
#     # 生成分析报告
#     report_path = os.path.join(output_dir, "analysis_report.txt")
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write("Individual Image HSV Analysis Report\n")
#         f.write("=" * 50 + "\n")
#         for image_name, paths in results.items():
#             f.write(f"\nImage: {image_name}\n")
#             f.write(f"  Donut Chart: {paths['donut_chart']}\n")
#             f.write(f"  H-S Radar Chart: {paths['hs_radar']}\n")
#             f.write(f"  H-V Radar Chart: {paths['hv_radar']}\n")
#             f.write(f"  H-S 3D Histogram: {paths['hs_3d_histogram']}\n")
    
#     print(f"\n[Complete] All images analyzed successfully!")
#     print(f"Report saved: {report_path}")
#     print(f"Output directory: {output_dir}")


# # ==========================================================
# # 原有的整体分析函数（保持不变）
# # ==========================================================
# def hsv_analysis(
#     txt_path="output/visual/group/group_color.txt",
#     output_dir="output1/hsv_final"
# ):
#     """原有的整体分析函数，保持不变"""
#     os.makedirs(output_dir, exist_ok=True)

#     HEX, RGB, RATIO = [], [], []

#     rgb_pattern = re.compile(r"np\.int64\((\d+)\)|(\d+)")

#     with open(txt_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue

#             parts = line.split()
#             hex_code = parts[0]

#             numbers = [int(m[0] or m[1]) for m in rgb_pattern.findall(line)]
#             rgb_vals = tuple(numbers[:3])

#             ratio_str = parts[-1].replace("%", "")
#             ratio_val = float(ratio_str) / 100.0

#             HEX.append(hex_code)
#             RGB.append(rgb_vals)
#             RATIO.append(ratio_val)

#     RGB = np.array(RGB, dtype=float)
#     weights = np.array(RATIO)

#     H_list, S_list, V_list, color_list = [], [], [], []

#     for rgb in RGB:
#         r, g, b = safe_rgb(rgb)
#         H, S, V = rgb_to_hsv_manual(r, g, b)

#         H_list.append(H)
#         S_list.append(S)
#         V_list.append(V)
#         color_list.append((r, g, b))

#     H_list = np.array(H_list)
#     S_list = np.array(S_list)
#     V_list = np.array(V_list)

#     order = np.argsort(H_list)
#     H_list = H_list[order]
#     S_list = S_list[order]
#     V_list = V_list[order]
#     weights = weights[order]
#     color_list = [color_list[i] for i in order]

#     plot_radar(
#         H_list, S_list, color_list, weights,
#         "H-S Color Distribution Radar Chart",
#         f"{output_dir}/HS_radar.png"
#     )

#     plot_radar(
#         H_list, V_list, color_list, weights,
#         "H-V Color Distribution Radar Chart",
#         f"{output_dir}/HV_radar.png"
#     )

#     print("[HSV] Group HSV analysis completed. Output directory:", output_dir)


# # ==========================================================
# # 主程序入口
# # ==========================================================
# if __name__ == "__main__":
#     # 可以根据需要选择运行整体分析或单张图片分析
#     print("Analysis Mode Selection:")
#     print("1. Group analysis (using original txt file)")
#     print("2. Individual image analysis (using colors.json file)")
    
#     choice = input("Please enter your choice (1 or 2): ").strip()
    
#     if choice == "1":
#         hsv_analysis()
#     elif choice == "2":
#         individual_hsv_analysis()
#     else:
#         print("Invalid choice, default to individual image analysis")
#         individual_hsv_analysis()