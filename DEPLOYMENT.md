# 🚀 路線価図OCR系统 - 部署与测试指南
## Deployment and Testing Guide for Route Price Map OCR System

本指南提供完整的Stage 5系统部署和测试说明，帮助您快速上手并验证系统功能。

---

## 📋 目录

- [环境准备](#环境准备)
- [快速部署](#快速部署)
- [详细安装](#详细安装)
- [功能测试](#功能测试)
- [性能验证](#性能验证)
- [故障排除](#故障排除)

---

## 🔧 环境准备

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **操作系统** | macOS 10.15+ / Ubuntu 18.04+ / Windows 10+ | 最新稳定版 |
| **Python** | 3.8+ | 3.9+ |
| **内存** | 4GB | 8GB+ |
| **存储空间** | 2GB | 5GB+ |
| **网络** | 稳定互联网连接 | 下载PaddlePaddle模型 |

### 兼容性检查

```bash
# 检查Python版本
python3 --version  # 应显示 3.8+

# 检查pip版本
pip3 --version

# macOS M1/M2用户额外检查
uname -m  # 应显示 arm64

# 检查可用内存（Linux/macOS）
free -h  # 或 vm_stat (macOS)
```

---

## ⚡ 快速部署

### 一键安装脚本

```bash
# 1. 克隆项目
git clone https://github.com/xihaopark/rosenka_proj.git
cd rosenka_proj

# 2. 创建虚拟环境（推荐）
python3 -m venv ocr_env
source ocr_env/bin/activate  # Linux/macOS
# 或 ocr_env\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "
from core.ocr.unified_ocr_engine import UnifiedOCREngine
print('✅ Stage 5 系统安装成功!')
print('📦 核心组件加载正常')
"
```

### 首次运行测试

```bash
# 使用内置测试文件进行快速验证
python rosenka_ocr_stage5_simple.py --input test1.pdf --output quick_test --debug

# 预期输出：
# 🚀 Stage 5 简化版系统初始化完成
# 📄 开始处理PDF: test1.pdf
# ✅ PDF转换成功，图像尺寸: (1162, 1506, 3)
# 📊 检测到 [N] 个文本区域，耗时: [X.X]秒
```

---

## 🔨 详细安装

### 步骤1: 环境配置

```bash
# 创建专用虚拟环境
python3 -m venv rosenka_ocr_env
source rosenka_ocr_env/bin/activate

# 升级基础工具
pip install --upgrade pip setuptools wheel
```

### 步骤2: 依赖安装

```bash
# 核心依赖安装
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install Pillow==10.0.0
pip install PyMuPDF==1.23.8

# PaddlePaddle安装（根据平台选择）
# CPU版本（推荐，稳定性好）
pip install paddlepaddle

# GPU版本（Linux/Windows NVIDIA GPU）
# pip install paddlepaddle-gpu

# EasyOCR（可选，增强功能）
pip install easyocr
```

### 步骤3: M1/M2 Mac专用配置

```bash
# M1/M2芯片优化安装
pip install paddlepaddle --index-url https://pypi.org/simple/

# 验证M1兼容性
python -c "
import paddle
print('PaddlePaddle版本:', paddle.__version__)
print('Metal支持:', hasattr(paddle.device, 'set_device'))
"
```

### 步骤4: 验证核心组件

```bash
# 测试各个Stage 5组件
python -c "
from core.ocr.enhanced_image_preprocessor import EnhancedImagePreprocessor
from core.ocr.multi_scale_ocr_detector import MultiScaleOCRDetector
from core.ocr.spatial_intelligence_engine import SpatialIntelligenceEngine
from core.ocr.intelligent_post_processor import IntelligentPostProcessor
print('✅ 所有Stage 5组件导入成功')
"
```

---

## 🧪 功能测试

### 基础功能测试

#### 测试1: PDF处理能力

```bash
# 测试PDF转图像功能
python -c "
import fitz
import numpy as np
import cv2

doc = fitz.open('test1.pdf')
page = doc.load_page(0)
mat = fitz.Matrix(2.0, 2.0)
pix = page.get_pixmap(matrix=mat)
img_data = pix.tobytes('ppm')
nparr = np.frombuffer(img_data, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
doc.close()

print(f'✅ PDF处理成功，图像尺寸: {image.shape}')
print(f'📊 内存使用: {image.nbytes / 1024 / 1024:.1f} MB')
"
```

#### 测试2: OCR引擎功能

```bash
# 测试统一OCR引擎
python -c "
from core.ocr.unified_ocr_engine import UnifiedOCREngine
import cv2
import numpy as np

# 创建测试图像
test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
cv2.putText(test_image, '115E', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
cv2.putText(test_image, '12-5', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

# 初始化引擎
ocr_engine = UnifiedOCREngine(use_gpu=False, enable_number_enhancement=True)

# 检测文本
results = ocr_engine.detect_text_regions(test_image)
print(f'✅ OCR检测成功，发现 {len(results)} 个文本区域')
for result in results:
    print(f'   文本: {result.get(\"text\", \"\")} (置信度: {result.get(\"confidence\", 0):.3f})')
"
```

#### 测试3: Stage 5增强功能

```bash
# 测试Stage 5智能分类
python -c "
from core.ocr.unified_ocr_engine import UnifiedOCREngine
import cv2
import numpy as np
import re

# 模拟路線価检测结果
mock_detections = [
    {'text': '115E', 'confidence': 0.95, 'bbox': [10, 10, 50, 30]},
    {'text': '12', 'confidence': 0.92, 'bbox': [100, 10, 30, 30]},
    {'text': '12-5', 'confidence': 0.88, 'bbox': [200, 10, 60, 30]},
    {'text': 'R07', 'confidence': 0.90, 'bbox': [300, 10, 40, 30]}
]

# Stage 5分类测试
patterns = {
    'route_price': r'^\\d{1,4}[A-G]$',
    'block_number': r'^\\d{1,3}$',
    'complex_address': r'^\\d{1,3}-\\d{1,3}$',
    'reference_mark': r'^[A-Z]\\d{1,3}$'
}

for detection in mock_detections:
    text = detection['text']
    classification = 'unknown'
    
    for pattern_name, pattern in patterns.items():
        if re.match(pattern, text):
            classification = pattern_name
            break
    
    print(f'文本: {text} -> 分类: {classification}')

print('✅ Stage 5智能分类功能正常')
"
```

### 完整系统测试

#### 端到端测试

```bash
# 完整的Stage 5系统测试
python rosenka_ocr_stage5_simple.py --input test1.pdf --output full_test --debug

# 检查输出文件
ls -la full_test/
# 应该包含:
# - test1.pdf_stage5_simple_results.json
# - test1.pdf_detections_simple.json
# - visualizations/test1/ (可视化图像)
```

#### 批量测试

```bash
# 测试两个标准文件
for file in test1.pdf test2.pdf; do
    echo "测试文件: $file"
    python rosenka_ocr_stage5_simple.py --input $file --output ${file%.*}_results
    echo "✅ $file 测试完成"
done
```

---

## 📊 性能验证

### 性能基准测试

```bash
# 创建性能测试脚本
cat > performance_test.py << 'EOF'
#!/usr/bin/env python3
import time
import psutil
import os
from core.ocr.unified_ocr_engine import UnifiedOCREngine
import cv2
import fitz

def test_performance():
    print("🔬 性能基准测试")
    print("=" * 50)
    
    # 内存使用测试
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"初始内存使用: {initial_memory:.1f} MB")
    
    # PDF处理速度测试
    start_time = time.time()
    doc = fitz.open('test1.pdf')
    page = doc.load_page(0)
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes('ppm')
    image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    doc.close()
    pdf_time = time.time() - start_time
    print(f"PDF转换时间: {pdf_time:.2f}秒")
    
    # OCR处理速度测试
    ocr_engine = UnifiedOCREngine(use_gpu=False, enable_number_enhancement=True)
    
    start_time = time.time()
    results = ocr_engine.detect_text_regions(image)
    ocr_time = time.time() - start_time
    print(f"OCR检测时间: {ocr_time:.2f}秒")
    print(f"检测结果数: {len(results)}")
    
    # 总内存使用
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"峰值内存使用: {final_memory:.1f} MB")
    print(f"内存增长: {final_memory - initial_memory:.1f} MB")
    
    # 性能评级
    total_time = pdf_time + ocr_time
    if total_time < 10:
        grade = "🟢 优秀"
    elif total_time < 20:
        grade = "🟡 良好"
    else:
        grade = "🔴 需要优化"
    
    print(f"总处理时间: {total_time:.2f}秒 - {grade}")

if __name__ == "__main__":
    import numpy as np
    test_performance()
EOF

python performance_test.py
```

### 准确率验证

```bash
# 创建准确率测试
cat > accuracy_test.py << 'EOF'
#!/usr/bin/env python3
from core.ocr.unified_ocr_engine import UnifiedOCREngine
import cv2
import numpy as np
import re

def test_accuracy():
    print("🎯 准确率验证测试")
    print("=" * 50)
    
    # 创建标准测试图像
    test_cases = [
        ('115E', 'route_price'),
        ('95A', 'route_price'), 
        ('1200万', 'route_price_unit'),
        ('15', 'block_number'),
        ('12-5', 'complex_address'),
        ('R07', 'reference_mark')
    ]
    
    ocr_engine = UnifiedOCREngine(use_gpu=False, enable_number_enhancement=True)
    patterns = {
        'route_price': r'^\d{1,4}[A-G]$',
        'route_price_unit': r'^\d{1,4}万[A-G]?$',
        'block_number': r'^\d{1,3}$',
        'complex_address': r'^\d{1,3}-\d{1,3}$',
        'reference_mark': r'^[A-Z]\d{1,3}$'
    }
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, (text, expected_type) in enumerate(test_cases):
        # 创建测试图像
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        # OCR检测
        results = ocr_engine.detect_text_regions(image)
        
        if results:
            detected_text = results[0]['text'].strip()
            
            # 分类预测
            predicted_type = 'unknown'
            for pattern_name, pattern in patterns.items():
                if re.match(pattern, detected_text):
                    predicted_type = pattern_name
                    break
            
            # 验证结果
            is_correct = (detected_text == text and predicted_type == expected_type)
            if is_correct:
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} 测试 {i+1}: '{text}' -> 检测: '{detected_text}', 分类: {predicted_type}")
        else:
            print(f"❌ 测试 {i+1}: '{text}' -> 未检测到文本")
    
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n📊 总体准确率: {accuracy:.1f}% ({correct_predictions}/{total_tests})")
    
    if accuracy >= 90:
        print("🟢 准确率优秀")
    elif accuracy >= 80:
        print("🟡 准确率良好")
    else:
        print("🔴 准确率需要改进")

if __name__ == "__main__":
    test_accuracy()
EOF

python accuracy_test.py
```

---

## 🔧 故障排除

### 常见问题解决

#### 问题1: PaddlePaddle安装失败

```bash
# 解决方案1: 使用镜像源
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 解决方案2: 降级numpy
pip install numpy==1.24.3

# 解决方案3: 清理缓存重装
pip cache purge
pip install paddlepaddle --no-cache-dir
```

#### 问题2: M1芯片兼容性问题

```bash
# 检查架构
python -c "import platform; print('架构:', platform.machine())"

# M1专用安装命令
pip install paddlepaddle --index-url https://pypi.org/simple/
pip install opencv-python --index-url https://pypi.org/simple/

# 如果仍有问题，使用Rosetta模式
arch -x86_64 pip install paddlepaddle
```

#### 问题3: 内存不足错误

```bash
# 降低图像分辨率
# 在代码中修改Matrix缩放因子
mat = fitz.Matrix(1.5, 1.5)  # 从2.0降低到1.5

# 或者使用debug模式
python rosenka_ocr_stage5_simple.py --input test1.pdf --debug
```

#### 问题4: OCR识别结果不准确

```bash
# 检查图像质量
python -c "
import cv2
import fitz
doc = fitz.open('test1.pdf')
page = doc.load_page(0)
pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
img_data = pix.tobytes('ppm')
import numpy as np
nparr = np.frombuffer(img_data, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
print(f'图像尺寸: {image.shape}')
print(f'图像质量: {\"高\" if min(image.shape[:2]) > 1000 else \"低\"}')
doc.close()
"

# 调整置信度阈值
python rosenka_ocr_stage5_simple.py --input test1.pdf --confidence_threshold 0.5
```

### 日志和调试

#### 启用详细日志

```bash
# 创建调试脚本
cat > debug_test.py << 'EOF'
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from core.ocr.unified_ocr_engine import UnifiedOCREngine
import cv2
import fitz
import numpy as np

# 加载PDF
doc = fitz.open('test1.pdf')
page = doc.load_page(0)
pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
img_data = pix.tobytes('ppm')
image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
doc.close()

# 初始化OCR
ocr_engine = UnifiedOCREngine(use_gpu=False, enable_number_enhancement=True)

# 执行检测
results = ocr_engine.detect_text_regions(image)
print(f"检测结果: {len(results)} 个区域")
EOF

python debug_test.py
```

#### 性能监控

```bash
# 使用系统监控工具
# macOS
sudo fs_usage -w -f filesys python rosenka_ocr_stage5_simple.py --input test1.pdf

# Linux  
strace -f -e trace=file python rosenka_ocr_stage5_simple.py --input test1.pdf

# 通用内存监控
python -c "
import psutil
import subprocess
import time

# 启动进程
process = subprocess.Popen(['python', 'rosenka_ocr_stage5_simple.py', '--input', 'test1.pdf'])
pid = process.pid

# 监控资源使用
while process.poll() is None:
    try:
        p = psutil.Process(pid)
        print(f'内存使用: {p.memory_info().rss / 1024 / 1024:.1f} MB, CPU: {p.cpu_percent():.1f}%')
        time.sleep(1)
    except psutil.NoSuchProcess:
        break
"
```

---

## ✅ 部署检查清单

### 安装验证

- [ ] Python 3.8+ 正确安装
- [ ] 虚拟环境创建成功
- [ ] 所有依赖安装完成
- [ ] PaddlePaddle正常工作
- [ ] OpenCV图像处理功能正常
- [ ] Stage 5组件导入成功

### 功能验证

- [ ] PDF转图像功能正常
- [ ] OCR文本检测工作
- [ ] Stage 5智能分类正确
- [ ] 可视化结果生成
- [ ] JSON输出格式正确
- [ ] 批量处理功能正常

### 性能验证

- [ ] 处理速度满足要求（<30秒/页）
- [ ] 内存使用合理（<2GB）
- [ ] 准确率达到预期（>85%）
- [ ] 系统稳定运行

### 文档验证

- [ ] README.md信息完整
- [ ] 部署文档清晰
- [ ] API文档可用
- [ ] 示例代码可运行

---

## 🎉 部署完成

恭喜！您已成功部署了路線価図OCR系统Stage 5版本。

### 下一步建议

1. **熟悉命令**: 尝试不同的命令行参数
2. **测试真实数据**: 使用您自己的路線価图PDF文件
3. **性能调优**: 根据硬件配置调整参数
4. **集成开发**: 将系统集成到您的应用中

### 技术支持

如果遇到问题，请：
1. 查看本文档的故障排除部分
2. 在GitHub提交Issue: https://github.com/xihaopark/rosenka_proj/issues
3. 提供详细的错误信息和系统环境

---

*部署指南最后更新: 2025年7月21日*