import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

from ..models.yolo_model import YOLOModel
from ..utils.config_loader import load_config
from ..reports.visualization import DetectionVisualizer
from ..reports.metrics import DetectionMetrics

import altair as alt

@st.cache_resource
def load_cached_model(config):
    """
    使用 cache_resource 缓存模型实例。
    Streamlit 会检测 config 是否变化，只有变化时才会重新加载。
    """
    print("Loading model...")  # 调试用，你会发现它只打印一次
    return YOLOModel(config)


class StreamlitApp:
    def __init__(self):
        self.setup_page_config()
        self.config = load_config()
        # 注意：这里我们不再在 __init__ 里强行加载模型，而是按需加载
        self.apply_custom_css()
        self.initialize_session_state()

    def get_model(self):
        """获取缓存的模型的辅助方法"""
        return load_cached_model(self.config['model'])

    def process_image(self, image_file):
        """处理单张图片"""
        # 使用缓存加载，非常快
        model = self.get_model()
        
        image = Image.open(image_file)
        image = np.array(image)
        
        # 这里的 predict 已经是优化过的方法
        detections = model.predict(image)

        visualizer = DetectionVisualizer(model.class_names) # 确保传入 class_names
        
        return visualizer.draw_detections(image, detections)



    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="AI的目标检测系统",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def apply_custom_css(self):
        """应用自定义CSS样式"""
        st.markdown("""
        <style>
        /* 主题颜色 */
        :root {
            --primary-bg: #FEFCE8;
            --secondary-bg: #FFFFFF;
            --accent-color: #F59E0B;
            --text-color: #1F2937;
            --card-bg: #FEF3C7;
        }
        
        /* 全局样式 */
        .stApp {
            background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
            color: var(--text-color);
        }
        
        /* 侧边栏样式 */
        .css-1d391kg {
            background-color: var(--card-bg);
        }
        
        /* 卡片样式 */
        .card {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }
        
        /* 标题样式 */
        h1 {
            background: linear-gradient(90deg, #1E88E5, #64B5F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 2rem;
        }
        
        h2, h3 {
            color: var(--text-color);
            font-weight: 500;
        }
        
        /* 按钮样式 */
        .stButton>button {
            background: linear-gradient(90deg, #1E88E5, #64B5F6);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 5px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
        }
        
        /* 指标卡片样式 */
        .metric-card {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E88E5;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            color: var(--text-color);
            font-size: 1rem;
            opacity: 0.8;
        }
        
        /* 进度条样式 */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #1E88E5, #64B5F6);
        }
        
        /* 选择器样式 */
        .stSelectbox > div > div {
            background-color: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* 图表样式 */
        .stPlot {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 1rem;
        }
        
        /* 动画效果 */
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(30, 136, 229, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(30, 136, 229, 0); }
            100% { box-shadow: 0 0 0 0 rgba(30, 136, 229, 0); }
        }
        
        .detection-active {
            animation: pulse 2s infinite;
        }
        
        /* 上传区域样式 */
        .uploadfile {
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .uploadfile:hover {
            border-color: var(--accent-color);
        }
        
        /* 表格样式 */
        .dataframe {
            background: var(--card-bg);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """初始化会话状态"""
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'metrics' not in st.session_state:
            st.session_state.metrics = DetectionMetrics()
        if 'running' not in st.session_state:
            st.session_state.running = False

    def run(self):
        """运行Streamlit应用"""
        self.render_header()
        
        # 创建标签页
        tabs = st.tabs([
            "🎥 视频目标检测",
            "🖼️ 图片分析",
            "📊 分析大屏"
        ])
        
        with tabs[0]:
            self.render_realtime_detection()
        with tabs[1]:
            self.render_image_detection()
        with tabs[2]:
            self.render_analytics()

    def render_header(self):
        """渲染页面头部"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1>AI的目标检测</h1>
            <p style="color: #64B5F6; font-size: 1.2rem;">
                目标检测及分析
            </p>
        </div>
        """, unsafe_allow_html=True)

    def render_metric_card(self, title, value, icon):
        """渲染指标卡片"""
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; color: #64B5F6; margin-bottom: 0.5rem;">
                {icon}
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{title}</div>
        </div>
        """, unsafe_allow_html=True)

    def render_realtime_detection(self):
        """渲染实时检测页面"""
        col1, col2 = st.columns([6, 4])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            video_placeholder = st.empty()
            video_placeholder.write("导入文件")
            # 添加状态指示器
            if st.session_state.running:
                st.markdown("""
                    <div style="text-align: center; color: #1E88E5;">
                        🔴 正在录制
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 控制界面")
            source = st.radio(
                "Select Input Source",
                ["📹 摄像头", "📁 视频文件"],
                key="source_select"
            )

            if source == "📹 摄像头":
                camera_id = st.selectbox("选择通道", [0, 1, 2])
                self.run_camera_detection(camera_id, video_placeholder)
            else:
                st.markdown('<div class="uploadfile">', unsafe_allow_html=True)
                video_file = st.file_uploader(
                    "拖拽文件到此",
                    type=['mp4', 'avi', 'mov']
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if video_file:
                    self.run_video_detection(video_file, video_placeholder)

            st.markdown('</div>', unsafe_allow_html=True)

    def render_image_detection(self):
        """渲染图片检测页面"""
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 图片分析")
        
        upload_col, preview_col = st.columns([3,7])
        
        with upload_col:
            st.markdown('<div class="uploadfile">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "拖拽文件到此",
                type=['jpg', 'jpeg', 'png']
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        if uploaded_file:
            image = Image.open(uploaded_file)
            preview_col.image(image, caption="Preview", use_container_width=True)
            
            if st.button("🔍 分析文件"):
                with st.spinner("分析中..."):
                    result_image = self.process_image(uploaded_file)
                    st.image(result_image, caption="Detection Result")
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_analytics(self):
        """渲染分析页面（使用 Altair 优化图表）"""
        if 'metrics' not in st.session_state or not st.session_state.metrics:
            st.info("暂无数据，请先运行检测。")
            return

        metrics = st.session_state.metrics.get_summary()
        
        # 显示关键指标卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总计帧数", metrics['total_frames'])
        with col2:
            # 🟢 修正：使用 'average_fps' 而不是 'fps'
            st.metric("平均帧率", f"{metrics['average_fps']:.1f} FPS")
        with col3:
            st.metric("检测目标", metrics['total_detections'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- 图表优化部分 ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 目标分类统计")
        
        counts = metrics['class_distribution']
        if counts:
            model = self.get_model()
            class_names = model.class_names if model else []
            
            # 数据转换
            named_counts = []
            for cls_id, count in counts.items():
                if class_names and 0 <= cls_id < len(class_names):
                    name = class_names[cls_id]
                else:
                    name = f"Class {cls_id}"
                named_counts.append({"类别": name, "数量": count})
            
            # 创建 DataFrame
            chart_data = pd.DataFrame(named_counts)
            
            if not chart_data.empty:
                # 使用 Altair 构建图表
                bars = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('数量', title='检测数量'),
                    y=alt.Y('类别', sort='-x', title=''),
                    color=alt.Color('类别', legend=None),
                    tooltip=['类别', '数量']
                )
                
                text = bars.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3
                ).encode(
                    text='数量'
                )
                
                final_chart = (bars + text).properties(height=300)
                st.altair_chart(final_chart, use_container_width=True)
            else:
                 st.info("暂无有效分类数据")
        else:
            st.info("暂无分类统计数据")
            
        st.markdown('</div>', unsafe_allow_html=True)


    def process_image(self, image_file):
        """处理单张图片"""
        if st.session_state.model is None:
            st.session_state.model = YOLOModel(self.config['model'])
        
        image = Image.open(image_file)
        image = np.array(image)
        
        detections = st.session_state.model.predict(image)
        visualizer = DetectionVisualizer()
        
        return visualizer.draw_detections(image, detections)

    def run_camera_detection(self, camera_id, placeholder):
        """运行摄像头检测"""
        cap = cv2.VideoCapture(camera_id)
        
        model = self.get_model()
        
        visualizer = DetectionVisualizer(model.class_names)
        # 创建两列布局用于开始和停止按钮
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("▶️ 开始")
        with col2:
            stop_button = st.button("⏹️ 结束")
            
        if start_button:
            st.session_state.running = True
            st.session_state.metrics.reset()

        if stop_button:
            st.session_state.running = False
            
        try:
            while st.session_state.running:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    st.warning("无法读取摄像头画面")
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                detections = model.predict(frame)

                process_time = time.time() - start_time
                if st.session_state.metrics:
                    st.session_state.metrics.update(detections, process_time)

                if detections is not None:
                    frame = visualizer.draw_detections(frame, detections)
                
                placeholder.image(frame)
                
        finally:
            cap.release()
            st.session_state.running = False

    def run_video_detection(self, video_file, placeholder):
        """运行视频文件检测"""
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        model = self.get_model()
        
        visualizer = DetectionVisualizer(model.class_names)

        if st.session_state.model is None:
            st.session_state.model = YOLOModel(self.config['model'])
    

        # 创建两列布局用于开始和停止按钮
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("▶️ 开始")
        with col2:
            stop_button = st.button("⏹️ 结束")
            
        if start_button:
            st.session_state.running = True
            st.session_state.metrics.reset()

        if stop_button:
            st.session_state.running = False
            
        try:
            while st.session_state.running:

                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = st.session_state.model.predict(frame)
                
                process_time = time.time() - start_time
                if st.session_state.metrics:
                    st.session_state.metrics.update(detections, process_time)

                if detections is not None:
                    frame = visualizer.draw_detections(frame, detections)
                
                placeholder.image(frame)
                
        finally:
            cap.release()
            os.unlink(tfile.name)
            st.session_state.running = False 