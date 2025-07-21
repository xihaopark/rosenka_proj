#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
env_setup_with_report.py
环境设置脚本 - 自动生成详细报告
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

class EnvironmentSetupReporter:
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'environments': {},
            'packages': {},
            'operations': [],
            'errors': [],
            'recommendations': []
        }
        self.start_time = time.time()
    
    def log_operation(self, operation, status, details="", error=""):
        """记录操作"""
        op_record = {
            'operation': operation,
            'status': status,
            'details': details,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.report['operations'].append(op_record)
        
        if error:
            self.report['errors'].append(f"{operation}: {error}")
        
        print(f"{'✅' if status == 'success' else '❌'} {operation}")
        if details:
            print(f"   {details}")
        if error:
            print(f"   错误: {error}")
    
    def run_command(self, cmd, description, critical=False):
        """运行命令并记录结果"""
        try:
            if isinstance(cmd, str):
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log_operation(description, 'success', result.stdout.strip()[:200])
                return True
            else:
                self.log_operation(description, 'failed', '', result.stderr.strip()[:200])
                if critical:
                    return False
                return True
        except subprocess.TimeoutExpired:
            self.log_operation(description, 'timeout', '', '命令执行超时')
            return False
        except Exception as e:
            self.log_operation(description, 'error', '', str(e))
            return False
    
    def collect_system_info(self):
        """收集系统信息"""
        print("📊 收集系统信息...")
        
        # Python信息
        self.report['system_info']['python_version'] = sys.version
        self.report['system_info']['python_executable'] = sys.executable
        self.report['system_info']['working_directory'] = str(Path.cwd())
        
        # 环境变量
        self.report['system_info']['conda_default_env'] = os.environ.get('CONDA_DEFAULT_ENV', 'None')
        self.report['system_info']['virtual_env'] = os.environ.get('VIRTUAL_ENV', 'None')
        self.report['system_info']['path'] = os.environ.get('PATH', '')[:500]  # 截断PATH
        
        # 检查虚拟环境状态
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.report['system_info']['in_virtual_env'] = True
            self.report['system_info']['virtual_env_path'] = sys.prefix
        else:
            self.report['system_info']['in_virtual_env'] = False
        
        # GPU信息
        gpu_info = self.check_gpu()
        self.report['system_info']['gpu'] = gpu_info
    
    def check_gpu(self):
        """检查GPU状态"""
        gpu_info = {'available': False}
        
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info['available'] = True
                gpu_info['nvidia_smi_output'] = result.stdout.strip()
        except:
            pass
        
        # 检查CUDA
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info['cuda_available'] = True
                gpu_info['cuda_version'] = result.stdout.strip()
        except:
            gpu_info['cuda_available'] = False
        
        return gpu_info
    
    def check_conda_environments(self):
        """检查conda环境"""
        print("🐍 检查Conda环境...")
        
        try:
            # 检查conda版本
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.report['environments']['conda_available'] = True
                self.report['environments']['conda_version'] = result.stdout.strip()
                
                # 列出环境
                result = subprocess.run(['conda', 'env', 'list', '--json'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    env_data = json.loads(result.stdout)
                    self.report['environments']['conda_envs'] = env_data['envs']
                    self.log_operation("检查conda环境", 'success', f"找到{len(env_data['envs'])}个环境")
                else:
                    self.log_operation("列出conda环境", 'failed', '', result.stderr)
            else:
                self.report['environments']['conda_available'] = False
                self.log_operation("检查conda", 'failed', '', result.stderr)
        except Exception as e:
            self.report['environments']['conda_available'] = False
            self.log_operation("检查conda", 'error', '', str(e))
    
    def check_current_packages(self):
        """检查当前已安装的包"""
        print("📦 检查已安装包...")
        
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                self.report['packages']['installed'] = packages
                self.log_operation("获取包列表", 'success', f"找到{len(packages)}个包")
            else:
                self.log_operation("获取包列表", 'failed', '', result.stderr)
        except Exception as e:
            self.log_operation("获取包列表", 'error', '', str(e))
        
        # 检查关键包
        key_packages = [
            'numpy', 'opencv-python', 'torch', 'paddleocr', 'easyocr', 
            'streamlit', 'pillow', 'pytesseract', 'rapidfuzz', 'transformers'
        ]
        
        missing_packages = []
        installed_versions = {}
        
        for package in key_packages:
            try:
                if package == 'opencv-python':
                    import cv2
                    installed_versions[package] = cv2.__version__
                elif package == 'pillow':
                    from PIL import Image
                    import PIL
                    installed_versions[package] = PIL.__version__
                else:
                    module = __import__(package.replace('-', '_'))
                    version = getattr(module, '__version__', 'unknown')
                    installed_versions[package] = version
            except ImportError:
                missing_packages.append(package)
        
        self.report['packages']['key_packages_installed'] = installed_versions
        self.report['packages']['missing_packages'] = missing_packages
    
    def create_conda_environment(self):
        """创建conda环境"""
        if not self.report['environments'].get('conda_available', False):
            self.log_operation("创建conda环境", 'skipped', '', 'Conda不可用')
            return False
        
        env_name = "rosenka_gpu"
        print(f"🔧 创建conda环境: {env_name}")
        
        # 检查环境是否已存在
        existing_envs = self.report['environments'].get('conda_envs', [])
        if any(env_name in env_path for env_path in existing_envs):
            self.log_operation(f"创建conda环境 {env_name}", 'skipped', '环境已存在')
            return True
        
        # 创建环境
        success = self.run_command(
            ['conda', 'create', '-n', env_name, 'python=3.10', '-y'],
            f"创建conda环境 {env_name}"
        )
        
        if success:
            # 安装基础包
            packages_to_install = [
                'numpy<2.0',
                'pandas', 
                'matplotlib',
                'jupyter',
                'opencv',
                'pillow'
            ]
            
            for package in packages_to_install:
                self.run_command(
                    ['conda', 'install', '-n', env_name, package, '-y'],
                    f"在{env_name}中安装 {package}"
                )
            
            # 使用pip安装特殊包
            pip_packages = [
                'streamlit',
                'PyMuPDF',
                'rapidfuzz',
                'jieba',
                'pytesseract'
            ]
            
            # GPU包
            if self.report['system_info']['gpu']['available']:
                pip_packages.extend([
                    'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
                    'paddlepaddle-gpu',
                    'paddleocr',
                    'easyocr'
                ])
            else:
                pip_packages.extend([
                    'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
                    'paddlepaddle',
                    'paddleocr',
                    'easyocr'
                ])
            
            for package in pip_packages:
                if '--index-url' in package:
                    # 特殊处理PyTorch安装
                    parts = package.split()
                    self.run_command(
                        ['conda', 'run', '-n', env_name, 'pip', 'install'] + parts,
                        f"在{env_name}中pip安装 {parts[0]}"
                    )
                else:
                    self.run_command(
                        ['conda', 'run', '-n', env_name, 'pip', 'install', package],
                        f"在{env_name}中pip安装 {package}"
                    )
        
        return success
    
    def fix_current_environment(self):
        """修复当前环境"""
        print("🔧 修复当前环境...")
        
        # 修复numpy兼容性问题
        self.run_command(
            ['pip', 'uninstall', 'opencv-python', 'opencv-contrib-python', 'opencv-python-headless', '-y'],
            "卸载冲突的OpenCV包"
        )
        
        self.run_command(
            ['pip', 'install', 'numpy<2.0', '--force-reinstall'],
            "降级NumPy到兼容版本"
        )
        
        self.run_command(
            ['pip', 'install', 'opencv-python-headless'],
            "重新安装OpenCV"
        )
        
        # 安装缺失的包
        missing = self.report['packages'].get('missing_packages', [])
        if missing:
            for package in missing:
                if package == 'torch':
                    if self.report['system_info']['gpu']['available']:
                        self.run_command(
                            ['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'],
                            "安装PyTorch GPU版本"
                        )
                    else:
                        self.run_command(
                            ['pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'],
                            "安装PyTorch CPU版本"
                        )
                elif package == 'paddleocr':
                    if self.report['system_info']['gpu']['available']:
                        self.run_command(['pip', 'install', 'paddlepaddle-gpu'], "安装PaddlePaddle GPU版本")
                    else:
                        self.run_command(['pip', 'install', 'paddlepaddle'], "安装PaddlePaddle CPU版本")
                    self.run_command(['pip', 'install', 'paddleocr'], "安装PaddleOCR")
                else:
                    self.run_command(['pip', 'install', package], f"安装 {package}")
    
    def test_installations(self):
        """测试安装结果"""
        print("🧪 测试安装结果...")
        
        test_results = {}
        
        # 测试关键包导入
        test_packages = [
            ('numpy', 'import numpy'),
            ('opencv', 'import cv2'),
            ('torch', 'import torch'),
            ('PIL', 'from PIL import Image'),
            ('streamlit', 'import streamlit'),
            ('rapidfuzz', 'import rapidfuzz'),
        ]
        
        for name, import_cmd in test_packages:
            try:
                exec(import_cmd)
                test_results[name] = 'success'
                self.log_operation(f"测试导入 {name}", 'success')
            except ImportError as e:
                test_results[name] = f'failed: {str(e)}'
                self.log_operation(f"测试导入 {name}", 'failed', '', str(e))
        
        # 测试OCR引擎
        try:
            from paddleocr import PaddleOCR
            test_results['paddleocr'] = 'success'
            self.log_operation("测试PaddleOCR", 'success')
        except Exception as e:
            test_results['paddleocr'] = f'failed: {str(e)}'
            self.log_operation("测试PaddleOCR", 'failed', '', str(e))
        
        try:
            import easyocr
            test_results['easyocr'] = 'success'
            self.log_operation("测试EasyOCR", 'success')
        except Exception as e:
            test_results['easyocr'] = f'failed: {str(e)}'
            self.log_operation("测试EasyOCR", 'failed', '', str(e))
        
        self.report['packages']['test_results'] = test_results
    
    def generate_recommendations(self):
        """生成建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        test_results = self.report['packages'].get('test_results', {})
        failed_tests = [name for name, result in test_results.items() if 'failed' in str(result)]
        
        if failed_tests:
            recommendations.append(f"需要修复以下包的导入问题: {', '.join(failed_tests)}")
        
        # 环境建议
        if self.report['environments'].get('conda_available', False):
            recommendations.append("推荐使用conda环境 'rosenka_gpu' 进行GPU加速任务")
        
        if self.report['system_info']['gpu']['available']:
            recommendations.append("检测到GPU，建议使用GPU加速版本的深度学习框架")
        else:
            recommendations.append("未检测到GPU，使用CPU版本的深度学习框架")
        
        # 项目启动建议
        if not failed_tests:
            recommendations.append("环境配置完成，可以启动项目: python launch_gpu_app.py")
        else:
            recommendations.append("需要先修复环境问题再启动项目")
        
        self.report['recommendations'] = recommendations
    
    def save_report(self):
        """保存报告"""
        self.report['execution_time'] = time.time() - self.start_time
        self.report['summary'] = {
            'total_operations': len(self.report['operations']),
            'successful_operations': len([op for op in self.report['operations'] if op['status'] == 'success']),
            'failed_operations': len([op for op in self.report['operations'] if op['status'] == 'failed']),
            'errors_count': len(self.report['errors'])
        }
        
        # 保存JSON报告
        report_file = f"environment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        # 生成可读的文本报告
        text_report = self.generate_text_report()
        text_file = f"environment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"\n📋 报告已保存:")
        print(f"   JSON格式: {report_file}")
        print(f"   文本格式: {text_file}")
        
        return report_file, text_file
    
    def generate_text_report(self):
        """生成文本格式报告"""
        report_lines = [
            "=" * 80,
            "🔧 Python环境设置报告",
            "=" * 80,
            f"生成时间: {self.report['timestamp']}",
            f"执行时长: {self.report['execution_time']:.2f}秒",
            "",
            "📊 系统信息:",
            "-" * 40,
            f"Python版本: {self.report['system_info']['python_version'].split()[0]}",
            f"Python路径: {self.report['system_info']['python_executable']}",
            f"工作目录: {self.report['system_info']['working_directory']}",
            f"虚拟环境: {'是' if self.report['system_info']['in_virtual_env'] else '否'}",
            f"Conda环境: {self.report['system_info']['conda_default_env']}",
            f"GPU可用: {'是' if self.report['system_info']['gpu']['available'] else '否'}",
            "",
            "🐍 Conda状态:",
            "-" * 40,
        ]
        
        if self.report['environments'].get('conda_available', False):
            report_lines.append(f"✅ Conda可用: {self.report['environments']['conda_version']}")
            envs = self.report['environments'].get('conda_envs', [])
            report_lines.append(f"   环境数量: {len(envs)}")
            for env in envs:
                env_name = os.path.basename(env)
                report_lines.append(f"   - {env_name}: {env}")
        else:
            report_lines.append("❌ Conda不可用")
        
        report_lines.extend([
            "",
            "📦 包状态:",
            "-" * 40,
        ])
        
        installed = self.report['packages'].get('key_packages_installed', {})
        missing = self.report['packages'].get('missing_packages', [])
        
        report_lines.append(f"✅ 已安装包 ({len(installed)}):")
        for pkg, version in installed.items():
            report_lines.append(f"   - {pkg}: {version}")
        
        if missing:
            report_lines.append(f"\n❌ 缺失包 ({len(missing)}):")
            for pkg in missing:
                report_lines.append(f"   - {pkg}")
        
        report_lines.extend([
            "",
            "🧪 测试结果:",
            "-" * 40,
        ])
        
        test_results = self.report['packages'].get('test_results', {})
        for name, result in test_results.items():
            status = "✅" if result == 'success' else "❌"
            report_lines.append(f"{status} {name}: {result}")
        
        report_lines.extend([
            "",
            "⚙️ 执行的操作:",
            "-" * 40,
        ])
        
        for op in self.report['operations']:
            status = "✅" if op['status'] == 'success' else "❌"
            report_lines.append(f"{status} {op['operation']}")
            if op['error']:
                report_lines.append(f"     错误: {op['error']}")
        
        if self.report['errors']:
            report_lines.extend([
                "",
                "❌ 错误汇总:",
                "-" * 40,
            ])
            for error in self.report['errors']:
                report_lines.append(f"   - {error}")
        
        report_lines.extend([
            "",
            "💡 建议:",
            "-" * 40,
        ])
        
        for rec in self.report['recommendations']:
            report_lines.append(f"   - {rec}")
        
        report_lines.extend([
            "",
            "=" * 80,
            f"报告结束 - 总计 {self.report['summary']['successful_operations']}/{self.report['summary']['total_operations']} 操作成功",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

def main():
    """主函数"""
    print("🚀 环境设置和报告生成器")
    print("=" * 60)
    print("此脚本将:")
    print("1. 检测当前环境状态")
    print("2. 创建/优化conda和venv环境")
    print("3. 安装必要的包")
    print("4. 测试安装结果")
    print("5. 生成详细报告")
    print("=" * 60)
    
    reporter = EnvironmentSetupReporter()
    
    try:
        # 执行所有步骤
        reporter.collect_system_info()
        reporter.check_conda_environments()
        reporter.check_current_packages()
        
        # 询问是否执行安装操作
        print("\n是否执行环境设置操作？(y/n): ", end="")
        response = input().lower().strip()
        
        if response == 'y':
            reporter.create_conda_environment()
            reporter.fix_current_environment()
        
        reporter.test_installations()
        reporter.generate_recommendations()
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        reporter.log_operation("用户中断", 'cancelled')
    except Exception as e:
        print(f"\n❌ 执行过程中出现异常: {e}")
        reporter.log_operation("执行异常", 'error', '', str(e))
    finally:
        # 无论如何都生成报告
        report_files = reporter.save_report()
        print(f"\n🎉 环境设置完成！请查看报告文件了解详细信息。")

if __name__ == "__main__":
    main() 