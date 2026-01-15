#! /usr/bin/python3
# -*- coding:utf-8 -*-
# @File:main.py
# @Author:B站-夜游神人
# @Time:2026/01/12 22:55:34

1.本项目主要用于图像色彩聚类与分析，将分析所得数据进行可视化！

2.运行代码前安装所需库文件
-requirement.txt
pip install -r requirements.txt

3.获取到代码后，将在当前目录创建数据集datasets文件夹
--datasets
    -001.png
    -002.png
    ……

注意：数据集图像需要进行抠图处理，并设置为白色背景才能适配本代码！
如果数据集进行了分类存放，在RGB分析阶段和HSV分析阶段需要修改文件夹：
        1、在RGB阶段修改  kmeans_individual.py中的input_dir="datasets/xxx",之后将输出文件夹重命名output_xxx;
        2、在HSV阶段修改  HSV_Analysis_3D.py中的相关代码：
                                    运行代码的文件夹修改入口：单个HSV分析在503行；整体HSV分析596行，请跳转修改！！！

运行main.py文件后，再运行HSV_Analysis_3D.py文件，结果保存在output3文件夹中！
    其中运行HSV_Analysis_3D.py文件需要进行交互选择
    “1”：进行HSV的群体分析；
    “2”：进行HSV的个体分析；

4.本项目仅用于学习交流使用！


注意：输出文件说明：
        在outPut文件夹下
        --output 
            - clustering_group              #群体聚类（二次聚类后，16类）输出的原始聚类数据，后期群体可视化绘图基于此文件；
            - clustering_individual         #个体聚类（首次聚类后，10类）输出的原始聚类数据，后期个体可视化绘图基于此文件；
            - hsv_final_group               #RGB转HSV后的群体可视化分析；
            - hsv_individual                #RGB转HSV后的个体可视化分析；
            - network                       #RGB色彩分析下的网络模型可视化；
            - visual                        #群体 / 个体 聚类可视化结果展示；
                - group
                - indicidual


