from .base_model import BaseModel
import torch
import numpy as np
import os
import logging  # 🟢 1. 补上缺失的导入
from ultralytics import YOLO

class YOLOModel(BaseModel):
    def __init__(self, config):
        """
        config 字典现在需要包含:
        - weights: 权重路径
        - type: 'v5', 'v8', 或 'v11'
        """
        super().__init__(config)
        
        # 🟢 2. 初始化 Logger (必须放在 load_model 之前)
        self.logger = logging.getLogger(__name__)
        
        self.model_type = config.get('type', 'v5')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        
        self.model = self.load_model()
        self.class_names = self._load_class_names()
        
    def load_model(self):
        weights_path = self.config['weights']
        # 现在这一行不会报错了
        self.logger.info(f"正在加载模型 [{self.model_type}]: {weights_path}...")

        try:
            if self.model_type == 'v5':
                # YOLOv5 使用 torch.hub 加载
                # 注意：确保 path 指向本地文件时使用 'custom'
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
                model.conf = self.confidence_threshold
                model.iou = self.iou_threshold
                return model
            
            elif self.model_type in ['v8', 'v11']:
                # YOLOv8 和 v11 使用 ultralytics 库加载
                model = YOLO(weights_path)
                return model
                
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
                
        except Exception as e:
            # 这里记录详细错误日志
            self.logger.error(f"模型加载失败详情: {e}")
            raise RuntimeError(f"模型加载失败: {e}")

    def predict(self, image):
        """统一输出格式为 numpy array: [[x1, y1, x2, y2, conf, cls], ...]"""
        if self.model_type == 'v5':
            with torch.no_grad():
                results = self.model(image)
            return results.xyxy[0].cpu().numpy()
            
        elif self.model_type in ['v8', 'v11']:
            # verbose=False 防止控制台刷屏
            results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
            
            result = results[0]
            if len(result.boxes) == 0:
                return np.array([])
            
            # result.boxes.data 已经是 [x1, y1, x2, y2, conf, cls] 格式
            return result.boxes.data.cpu().numpy()
            
        return np.array([])

    def _load_class_names(self):
        # 优先使用模型内置的 names
        if hasattr(self.model, 'names'):
            return self.model.names
        
        # v5 hub 模型有时是 module.names
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
            return self.model.module.names
            
        return super()._load_class_names()