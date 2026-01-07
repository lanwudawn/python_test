import os
import sys
import warnings

# 提前导入必要库，但核心逻辑全部移入main块
import streamlit as st

def main():
    # ========== 所有初始化操作移入main块（关键） ==========
    # 1. 添加项目根目录到 Python 路径（上下文内执行）
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)

    # 2. 调试打印（仅在主进程执行，避免上下文外输出）
    print("Python path:", sys.path)
    print("Project root:", project_root)

    # 3. 屏蔽警告（同时覆盖Streamlit的ScriptRunContext警告）
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")
    warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

    # 4. 延迟导入（避免导入时触发上下文问题）
    from src.my_app.frontend.app import StreamlitApp

    # 5. 初始化并运行App（确保在Streamlit上下文内）
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    # 所有逻辑通过main函数执行，确保Streamlit上下文完整初始化
    main()