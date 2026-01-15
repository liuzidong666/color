#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @File:main.py
# @Author:B站-夜游神人
# @Time:2026/01/12 22:55:34

from kmeans_individual import extract_individual_colors
from visualize_colors_circle import visualize_individual_colors,visualize_group_colors

from HSV_Analysis_3D import individual_hsv_analysis,hsv_analysis

from kmeans_group import extract_group_colors
from color_network import build_color_network

if __name__ == "__main__":


    print("==== 1. 一次聚类（服饰色彩） ====")
    extract_individual_colors(K=10)

    print("==== 2. 一次聚类可视化 ====")
    visualize_individual_colors()

    print("==== 3. 二次聚类 ====")
    extract_group_colors(K=16)

    print("==== 4. 二次聚类可视化 ====")
    visualize_group_colors()

    print("==== 5. HSV 分析 ====")
    print("\n本部分前往单独执行HSV_Analysis_3D.py, 此处跳过联合执行！！！")

    print("==== 6. 构建色彩网络模型 ====")
    build_color_network()

    print("\n项目全部执行完成,结果保存至output文件夹中,请前往查看！！！")