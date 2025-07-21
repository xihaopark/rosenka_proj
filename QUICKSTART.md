# ⚡ 快速启动指南
## Quick Start Guide for Route Price Map OCR System

5分钟快速上手路線価図OCR系统Stage 5版本。

---

## 🚀 一分钟安装

```bash
# 1. 克隆并进入项目
git clone https://github.com/xihaopark/rosenka_proj.git && cd rosenka_proj

# 2. 安装依赖
pip install -r requirements.txt

# 3. 立即测试
python rosenka_ocr_stage5_simple.py --input test1.pdf --output quick_test
```

## 🎯 立即测试

### 基础测试（1分钟）

```bash
# 测试单个文件
python rosenka_ocr_stage5_simple.py --input test1.pdf --debug

# 预期输出：
# ✅ PDF转换成功，图像尺寸: (1162, 1506, 3)
# 📊 检测到 [N] 个文本区域，耗时: [X]秒
# 💾 结果已保存: test1.pdf_stage5_simple_results.json
```

### 批量测试（2分钟）

```bash
# 测试两个标准文件
python rosenka_ocr_stage5_simple.py --input test1.pdf --output test1_results
python rosenka_ocr_stage5_simple.py --input test2.pdf --output test2_results

# 查看结果
ls test*_results/
```

## 📊 查看结果

### JSON结果文件

```bash
# 查看检测结果
cat test1_results/test1.pdf_detections_simple.json | head -20

# 统计检测数量
python -c "
import json
with open('test1_results/test1.pdf_detections_simple.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    total = data['summary']['total_detections']
    target = data['summary']['type_distribution']
    print(f'✅ 总检测: {total}')
    print(f'📊 类型分布: {target}')
"
```

### 可视化结果

```bash
# 查看生成的可视化图像
ls test1_results/visualizations/*/
# 包含: page_01_stage5_visualization.jpg

# 在macOS中打开查看
open test1_results/visualizations/*/page_01_stage5_visualization.jpg
```

## 🔧 常用命令

### 基本使用

```bash
# 最简单的使用
python rosenka_ocr_stage5_simple.py --input your_file.pdf

# 指定输出目录
python rosenka_ocr_stage5_simple.py --input your_file.pdf --output my_results

# 启用调试信息
python rosenka_ocr_stage5_simple.py --input your_file.pdf --debug

# GPU加速（如果支持）
python rosenka_ocr_stage5_simple.py --input your_file.pdf --gpu
```

### 高级选项

```bash
# 完整版系统
python rosenka_ocr_stage5.py --input your_file.pdf --config custom.yaml

# 批量处理（v4兼容）
python batch_processor_v4.py --input_dir pdf_folder/ --use_gpu
```

## 📋 快速检查清单

### 安装检查（30秒）

```bash
# 检查Python版本
python --version  # 应显示 3.8+

# 检查核心组件
python -c "from core.ocr.unified_ocr_engine import UnifiedOCREngine; print('✅ 安装成功')"

# 检查依赖
python -c "import cv2, numpy, pandas, fitz; print('✅ 依赖完整')"
```

### 功能检查（1分钟）

```bash
# 快速功能测试
python -c "
from core.ocr.unified_ocr_engine import UnifiedOCREngine
import numpy as np
import cv2

# 创建测试图像
img = np.ones((100, 200, 3), dtype=np.uint8) * 255
cv2.putText(img, '115E', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# OCR测试
ocr = UnifiedOCREngine(use_gpu=False)
results = ocr.detect_text_regions(img)
print(f'✅ OCR正常: 检测到{len(results)}个区域')
"
```

## 🎯 使用场景

### 场景1: 快速识别单个PDF

```bash
# 处理单个路線価图文件
python rosenka_ocr_stage5_simple.py --input route_price_map.pdf --output results

# 查看识别到的路線価
grep -o '"text":[^,]*' results/route_price_map.pdf_detections_simple.json | head -10
```

### 场景2: 批量处理多个文件

```bash
# 创建批处理脚本
cat > batch_process.sh << 'EOF'
#!/bin/bash
for pdf in *.pdf; do
    echo "处理: $pdf"
    python rosenka_ocr_stage5_simple.py --input "$pdf" --output "${pdf%.*}_results"
done
echo "✅ 批量处理完成"
EOF

chmod +x batch_process.sh
./batch_process.sh
```

### 场景3: 集成到Python项目

```python
# 在您的Python代码中使用
from core.ocr.unified_ocr_engine import UnifiedOCREngine
import cv2

# 初始化OCR引擎
ocr_engine = UnifiedOCREngine(
    use_gpu=False,
    enable_number_enhancement=True,
    confidence_threshold=0.3
)

# 处理图像
image = cv2.imread('route_map.jpg')
results = ocr_engine.detect_text_regions(image)

# 筛选路線価格式
route_prices = []
for result in results:
    text = result['text']
    if any(text.endswith(suffix) for suffix in ['A', 'B', 'C', 'D', 'E', 'F', 'G']):
        route_prices.append(text)

print(f"识别到的路線価: {route_prices}")
```

## 🔧 故障快速修复

### 问题1: 安装失败

```bash
# 创建虚拟环境
python -m venv ocr_env
source ocr_env/bin/activate  # Linux/macOS
# 或 ocr_env\Scripts\activate  # Windows

# 重新安装
pip install --upgrade pip
pip install -r requirements.txt
```

### 问题2: 识别结果不准确

```bash
# 降低置信度阈值
python rosenka_ocr_stage5_simple.py --input test1.pdf --confidence_threshold 0.2

# 启用调试模式查看详情
python rosenka_ocr_stage5_simple.py --input test1.pdf --debug
```

### 问题3: 处理速度慢

```bash
# 降低图像分辨率（在代码中修改）
# 将Matrix(2.0, 2.0)改为Matrix(1.5, 1.5)

# 或者使用较小的测试文件
python rosenka_ocr_stage5_simple.py --input test2.pdf  # 通常更小
```

## 📚 下一步学习

### 深入了解

1. **阅读完整文档**: [README.md](README.md)
2. **部署指南**: [DEPLOYMENT.md](DEPLOYMENT.md)
3. **Stage 5技术文档**: [Stage5_工程计划书.md](Stage5_工程计划书.md)

### 扩展使用

1. **自定义配置**: 修改confidence_threshold等参数
2. **集成开发**: 将OCR功能集成到您的应用
3. **性能优化**: 根据硬件配置调整处理参数

### 获取帮助

1. **GitHub Issues**: https://github.com/xihaopark/rosenka_proj/issues
2. **查看示例**: 项目中的test_*.py文件
3. **阅读源码**: core/ocr/目录下的组件代码

---

## 🎉 恭喜！

您已经成功启动了路線価図OCR系统。现在可以：

- ✅ 处理真实的路線価图PDF文件
- ✅ 获得高精度的文字识别结果  
- ✅ 享受Stage 5的智能分析功能
- ✅ 使用可视化结果验证准确性

开始探索更多功能吧！

---

*快速启动指南 - 让您5分钟上手路線価图OCR*  
*最后更新: 2025年7月21日*