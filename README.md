# 🎯 YOLO 多模型目标检测系统 (YOLO Multi-Model Detection System)

这是一个集成 **Web 可视化界面** 与 **命令行 (CLI)** 的综合目标检测项目。项目基于 PyTorch 和 Ultralytics 构建，支持 **YOLOv5**、**YOLOv8** 和 **YOLOv11** 多种模型版本的无缝切换与性能对比。

## ✨ 主要特性

* **多模型支持**：
    * **YOLOv5** (通过 `torch.hub` 加载)
    * **YOLOv8** & **YOLOv11** (通过 `ultralytics` 加载)
* **Web 可视化大屏 (Streamlit)**：
    * **实时检测**：支持调用摄像头或上传视频文件进行推理。
    * **图片分析**：单张图片的目标检测与可视化。
    * **数据看板**：实时显示 FPS、总检测数、类别分布统计。
    * **模型竞技场**：支持不同模型运行记录的 FPS 与检出数量对比。
* **命令行工具 (CLI)**：
    * 支持批量处理、自定义配置、无头模式运行。
    * 参数化控制输入源、计算设备及结果保存。

## 📂 目录结构

```text
.
├── config/
│   ├── config.yaml          # 核心配置文件
│   └── coco_classes.txt     # 类别名称文件
├── src/
│   └── my_app/
│       ├── frontend/        # Streamlit Web 前端代码
│       ├── models/          # 模型加载与推理封装 (v5/v8/v11)
│       ├── reports/         # 评估指标与可视化绘图
│       ├── utils/           # 工具库 (日志、参数解析等)
│       └── videos/          # 视频流处理逻辑
├── weights/                 # 模型权重存放目录 (需自行下载)
├── main.py                  # CLI 程序入口
├── run_frontend.py          # Web 程序启动入口
└── requirements.txt         # 项目依赖清单
