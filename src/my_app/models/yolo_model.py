from .base_model import BaseModel
import torch
import os

class YOLOModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.model = self.load_model()
        self.class_names = self._load_class_names()
        
    def load_model(self):
        # 增加对本地权重的检查，避免不必要的下载尝试
        weights_path = self.config['weights']
        source = 'custom' if os.path.exists(weights_path) else 'ultralytics/yolov5'
        
        # 即使是 hub load 也建议捕获异常
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        except Exception as e:
             # Fallback logic could go here
            raise RuntimeError(f"Failed to load model from {weights_path}: {e}")

        model.conf = self.confidence_threshold
        model.iou = self.iou_threshold
        self.to_device()
        return model
        
    def predict(self, image):
        # 性能优化：禁用梯度计算
        with torch.no_grad():
            results = self.model(image)
        return results.xyxy[0].cpu().numpy()
    
    def _load_class_names(self):
        """加载类别名称，使用UTF-8编码"""
        try:
            class_names_file = self.config.get('class_names_file')
            if class_names_file and os.path.exists(class_names_file):
                with open(class_names_file, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f.readlines()]
            return None
        except Exception as e:
            print(f"Error loading class names: {e}")
            return None 