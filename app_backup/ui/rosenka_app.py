#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rosenka_app.py
路線価図查询系统 - 统一入口应用
集成API服务和Web界面，提供单一启动点
"""

import os
import sys
import time
import signal
import threading
import subprocess
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import uvicorn
    from fastapi import FastAPI
    import streamlit as st
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: pip install fastapi uvicorn streamlit")
    sys.exit(1)

class RosenkaApp:
    """路線価図统一应用"""
    
    def __init__(self):
        self.api_process = None
        self.web_process = None
        self.api_port = 8000
        self.web_port = 8501
        self.api_url = f"http://127.0.0.1:{self.api_port}"
        self.web_url = f"http://127.0.0.1:{self.web_port}"
        
    def start_api_service(self):
        """启动API服务"""
        print(f"🚀 启动API服务: {self.api_url}")
        
        # 导入API服务
        from simple_rosenka_service import app
        
        # 启动API服务
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=self.api_port,
            log_level="info",
            reload=False
        )
        server = uvicorn.Server(config)
        
        # 在新线程中运行
        def run_api():
            try:
                server.run()
            except Exception as e:
                print(f"❌ API服务启动失败: {e}")
        
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # 等待服务启动
        time.sleep(3)
        return api_thread
    
    def start_web_interface(self):
        """启动Web界面"""
        print(f"🌐 启动Web界面: {self.web_url}")
        
        # 设置环境变量
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '127.0.0.1'
        os.environ['STREAMLIT_SERVER_PORT'] = str(self.web_port)
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
        
        # 启动Streamlit
        try:
            import streamlit.web.cli as stcli
            sys.argv = [
                "streamlit", "run", "rosenka_web.py",
                "--server.address", "127.0.0.1",
                "--server.port", str(self.web_port),
                "--server.headless", "true"
            ]
            stcli.main()
        except Exception as e:
            print(f"❌ Web界面启动失败: {e}")
    
    def check_service_health(self, url, service_name):
        """检查服务健康状态"""
        import requests
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} 运行正常")
                return True
        except:
            pass
        print(f"❌ {service_name} 未响应")
        return False
    
    def run(self, mode="both"):
        """运行应用"""
        print("🗾 路線価図查询系统启动中...")
        print("=" * 50)
        
        # 检查虚拟环境
        if not os.path.exists("venv_simple"):
            print("❌ 虚拟环境不存在，请先运行: python setup_venv.py")
            return
        
        # 激活虚拟环境
        venv_python = "venv_simple/bin/python"
        if not os.path.exists(venv_python):
            print("❌ 虚拟环境Python不存在")
            return
        
        print(f"🐍 使用Python: {venv_python}")
        
        if mode in ["api", "both"]:
            # 启动API服务
            api_thread = self.start_api_service()
            
            # 等待API服务启动
            print("⏳ 等待API服务启动...")
            for i in range(10):
                if self.check_service_health(self.api_url, "API服务"):
                    break
                time.sleep(1)
            else:
                print("❌ API服务启动超时")
                return
        
        if mode in ["web", "both"]:
            # 启动Web界面
            web_thread = threading.Thread(target=self.start_web_interface, daemon=True)
            web_thread.start()
            
            # 等待Web界面启动
            print("⏳ 等待Web界面启动...")
            time.sleep(5)
        
        print("\n🎉 服务启动完成!")
        print("=" * 50)
        if mode in ["api", "both"]:
            print(f"📡 API服务: {self.api_url}")
            print(f"📖 API文档: {self.api_url}/docs")
        if mode in ["web", "both"]:
            print(f"🌐 Web界面: {self.web_url}")
        print("💡 按 Ctrl+C 停止所有服务")
        print("=" * 50)
        
        try:
            # 保持运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 正在停止服务...")
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理资源...")
        
        # 停止所有相关进程
        try:
            subprocess.run(["pkill", "-f", "rosenka"], check=False)
            subprocess.run(["pkill", "-f", "streamlit"], check=False)
            subprocess.run(["pkill", "-f", "uvicorn"], check=False)
        except:
            pass
        
        print("✅ 服务已停止")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="路線価図查询系统")
    parser.add_argument(
        "--mode", 
        choices=["api", "web", "both"], 
        default="both",
        help="运行模式: api(仅API), web(仅Web), both(API+Web)"
    )
    parser.add_argument(
        "--api-port", 
        type=int, 
        default=8000,
        help="API服务端口"
    )
    parser.add_argument(
        "--web-port", 
        type=int, 
        default=8501,
        help="Web界面端口"
    )
    
    args = parser.parse_args()
    
    # 创建应用实例
    app = RosenkaApp()
    app.api_port = args.api_port
    app.web_port = args.web_port
    app.api_url = f"http://127.0.0.1:{app.api_port}"
    app.web_url = f"http://127.0.0.1:{app.web_port}"
    
    # 注册信号处理器
    def signal_handler(signum, frame):
        print("\n🛑 收到停止信号...")
        app.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行应用
    app.run(args.mode)

if __name__ == "__main__":
    main() 