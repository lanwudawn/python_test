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
import altair as alt

from ..models.yolo_model import YOLOModel
from ..utils.config_loader import load_config
from ..reports.visualization import DetectionVisualizer
from ..reports.metrics import DetectionMetrics

# -------------------------------------------------------------------------
# 全局缓存加载函数
# -------------------------------------------------------------------------
@st.cache_resource
def load_cached_model(model_name, model_type, weights_path):
    """
    使用 cache_resource 缓存模型实例。
    Streamlit 会检测参数是否变化，只有变化时才会重新加载。
    """
    # 构造 config 字典传给 YOLOModel
    config = {
        'name': model_name,
        'type': model_type,
        'weights': weights_path,
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'classes_path': 'config/coco_classes.txt' # 备用
    }
    print(f"Loading model: {model_name} ({model_type})...")
    return YOLOModel(config)


# -------------------------------------------------------------------------
# Streamlit 主程序类
# -------------------------------------------------------------------------
class StreamlitApp:
    def __init__(self):
        self.setup_page_config()
        self.config = load_config()
        self.apply_custom_css()
        self.initialize_session_state()

    def get_model(self):
        """根据 Sidebar 的选择获取模型"""
        return load_cached_model(
            self.current_model_name, 
            self.current_model_config['type'], 
            self.current_model_config['path']
        )

    # -------------------------------------------------------------------------
    # 页面配置与样式
    # -------------------------------------------------------------------------
    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="基于YOLO的目标检测系统",
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
        section[data-testid="stSidebar"] {
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
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
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
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """初始化会话状态"""
        if 'metrics' not in st.session_state:
            st.session_state.metrics = DetectionMetrics()

        if 'running' not in st.session_state:
            st.session_state.running = False

        if 'model_history' not in st.session_state:
            st.session_state.model_history = {}

    def render_header(self):
        """渲染页面头部"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1>AI 目标检测系统</h1>
            <p style="color: #64B5F6; font-size: 1.2rem;">
                基于 YOLOv5/v8/v11 的多模型实时检测平台
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 主运行逻辑
    # -------------------------------------------------------------------------
    def run(self):
        """运行Streamlit应用"""
        self.render_header()
        
        st.sidebar.title("🛠️ 模型设置")

        # 🟢 仅保留 v5, v8, v11
        model_options = {
            "YOLOv5 (快速)": {"type": "v5", "path": "weights/yolov5s.pt"},
            "YOLOv8 (平衡)": {"type": "v8", "path": "weights/yolov8n.pt"},
            "YOLOv11 (进阶)": {"type": "v11", "path": "weights/yolo11n.pt"},
        }

        selected_model_name = st.sidebar.selectbox(
            "选择检测模型", 
            list(model_options.keys())
        )

        self.current_model_config = model_options[selected_model_name]
        self.current_model_name = selected_model_name

        st.sidebar.title("导航")

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

    # -------------------------------------------------------------------------
    # 核心检测功能函数
    # -------------------------------------------------------------------------
    def process_image(self, image_file):
        """处理单张图片"""
        # 🟢 1. 获取模型 (使用新的缓存接口)
        model = self.get_model()
        
        try:
            # 🟢 2. 读取图片并强制转换为 RGB
            # .convert('RGB') 会自动把 4通道(RGBA) 转成 3通道(RGB)，丢弃透明度
            image = Image.open(image_file).convert('RGB')
            image = np.array(image)
            
            # 🟢 3. 使用 model 实例预测
            detections = model.predict(image)
            
            # 🟢 4. 可视化 (传入中文类别)
            visualizer = DetectionVisualizer(model.class_names)
            
            return visualizer.draw_detections(image, detections)
            
        except Exception as e:
            st.error(f"图片处理失败: {e}")
            return None

    def run_camera_detection(self, camera_id, placeholder):
        """运行摄像头检测"""
        cap = cv2.VideoCapture(camera_id)
        
        # 获取模型
        model = self.get_model()
        
        # 初始化可视化工具
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
                
                # 执行预测
                detections = model.predict(frame)

                # 更新指标
                process_time = time.time() - start_time
                if st.session_state.metrics:
                    st.session_state.metrics.update(detections, process_time)

                if detections is not None:
                    frame = visualizer.draw_detections(frame, detections)
                
                placeholder.image(frame)

            # 循环结束后保存数据
            if st.session_state.metrics and st.session_state.metrics.total_frames > 0:
                summary = st.session_state.metrics.get_summary()
                
                # 保存到历史记录中
                st.session_state.model_history[self.current_model_name] = {
                    "fps": summary['average_fps'],
                    "total_detections": summary['total_detections'],
                    "frames": summary['total_frames'],
                }
                # 显示提示
                st.toast(f"✅ {self.current_model_name} 测试数据已保存！")    

        finally:
            cap.release()
            st.session_state.running = False

    def run_video_detection(self, video_file, placeholder):
        """运行视频文件检测"""
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        # 🟢 获取模型 (统一接口)
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
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 🟢 使用 model 实例预测
                detections = model.predict(frame)
                
                process_time = time.time() - start_time
                if st.session_state.metrics:
                    st.session_state.metrics.update(detections, process_time)

                if detections is not None:
                    frame = visualizer.draw_detections(frame, detections)
                
                placeholder.image(frame)
            
            # 循环结束后保存数据
            if st.session_state.metrics and st.session_state.metrics.total_frames > 0:
                summary = st.session_state.metrics.get_summary()
                
                st.session_state.model_history[self.current_model_name] = {
                    "fps": summary['average_fps'],
                    "total_detections": summary['total_detections'],
                    "frames": summary['total_frames'],
                }
                st.toast(f"✅ {self.current_model_name} 测试数据已保存！")

        finally:
            cap.release()
            try:
                os.unlink(tfile.name)
            except:
                pass
            st.session_state.running = False

    # -------------------------------------------------------------------------
    # 页面渲染函数
    # -------------------------------------------------------------------------
    def render_realtime_detection(self):
        """渲染实时检测页面"""
        col1, col2 = st.columns([6, 4])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            video_placeholder = st.empty()
            video_placeholder.write("等待输入...")
            
            # 添加状态指示器
            if st.session_state.running:
                st.markdown("""
                    <div style="text-align: center; color: #1E88E5; margin-top: 10px;">
                        🔴 正在检测中...
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎮 控制台")
            source = st.radio(
                "选择输入源",
                ["📹 摄像头", "📁 视频文件"],
                key="source_select"
            )

            if source == "📹 摄像头":
                camera_id = st.selectbox("选择设备 ID", [0, 1, 2])
                st.info("确保摄像头未被其他应用占用")
                self.run_camera_detection(camera_id, video_placeholder)
            else:
                st.markdown('<div class="uploadfile">', unsafe_allow_html=True)
                video_file = st.file_uploader(
                    "拖拽视频文件到此",
                    type=['mp4', 'avi', 'mov']
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if video_file:
                    self.run_video_detection(video_file, video_placeholder)

            st.markdown('</div>', unsafe_allow_html=True)

    def render_image_detection(self):
        """渲染图片检测页面"""
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ 图片分析")
        
        upload_col, preview_col = st.columns([3,7])
        
        with upload_col:
            st.markdown('<div class="uploadfile">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "拖拽图片到此",
                type=['jpg', 'jpeg', 'png']
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        if uploaded_file:
            image = Image.open(uploaded_file)
            preview_col.image(image, caption="原始图片", use_container_width=True)
            
            if st.button("🔍 开始分析"):
                with st.spinner("AI 正在识别中..."):
                    result_image = self.process_image(uploaded_file)
                    if result_image is not None:
                        st.image(result_image, caption="检测结果", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_analytics(self):
        """渲染分析页面：包含当前会话详情 + 模型竞技场对比"""
        st.title("📊 分析大屏")
        
        # 1️⃣ 第一部分：当前模型会话分析
        current_name = getattr(self, 'current_model_name', '未选择模型')
        st.subheader(f"📍 当前会话: {current_name}")
        
        if 'metrics' in st.session_state and st.session_state.metrics:
            metrics = st.session_state.metrics.get_summary()
            
            # A. 关键指标卡片
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总计帧数", metrics['total_frames'])
            with col2:
                st.metric("平均帧率", f"{metrics['average_fps']:.1f} FPS")
            with col3:
                st.metric("检测目标", metrics['total_detections'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # B. 类别分布图
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🧬 目标类别分布 (当前)")
            
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
                
                chart_data = pd.DataFrame(named_counts)
                
                if not chart_data.empty:
                    # 使用 Altair 画横向条形图
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

        else:
            st.info("👆 请先在左侧选择模型，并运行【视频】或【摄像头】检测以生成数据。")

        st.markdown("---")
        
        # 2️⃣ 第二部分：模型性能竞技场 (对比分析)
        st.subheader("🏆 模型性能竞技场")
        st.caption("不同模型在当前运行期间的历史数据对比")
        
        if 'model_history' in st.session_state and len(st.session_state.model_history) > 0:
            history = st.session_state.model_history
            
            comp_data = []
            for name, data in history.items():
                row = data.copy()
                row['Model'] = name
                comp_data.append(row)
            
            df_comp = pd.DataFrame(comp_data)
            
            # C. 数据表格展示
            with st.expander("查看详细对比数据", expanded=True):
                cols = ['Model', 'fps', 'total_detections', 'frames']
                display_cols = [c for c in cols if c in df_comp.columns]
                st.dataframe(
                    df_comp[display_cols].style.format({'fps': "{:.2f}"}), 
                    use_container_width=True
                )
            
            # D. 可视化对比图表
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### 🚀 推理速度 (FPS)")
                chart_fps = alt.Chart(df_comp).mark_bar().encode(
                    x=alt.X('Model', title='模型名称', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('fps', title='帧率 (越高越好)'),
                    color=alt.Color('Model', legend=None),
                    tooltip=['Model', alt.Tooltip('fps', format='.1f')]
                ).properties(height=300)
                st.altair_chart(chart_fps, use_container_width=True)
                
            with c2:
                st.markdown("#### 🎯 累计检出数量")
                chart_count = alt.Chart(df_comp).mark_bar().encode(
                    x=alt.X('Model', title='模型名称', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('total_detections', title='检出总数'),
                    color=alt.Color('Model', legend=None),
                    tooltip=['Model', 'total_detections']
                ).properties(height=300)
                st.altair_chart(chart_count, use_container_width=True)
                
            if st.button("🗑️ 清空对比历史"):
                st.session_state.model_history = {}
                st.rerun()
                
        else:
            st.info("💡 **如何进行对比？**\n\n"
                    "1. 在左侧选择一个模型（如 YOLOv5），运行检测，然后停止。\n"
                    "2. 切换另一个模型（如 YOLOv8），再次运行检测。\n"
                    "3. 数据将自动汇聚于此进行 PK！")