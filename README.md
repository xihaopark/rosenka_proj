# 路線価图智能查询与下载系统

[![GitHub](https://img.shields.io/badge/GitHub-rosenka_proj-blue?logo=github)](https://github.com/xihaopark/rosenka_proj)

## 项目简介

本项目为日本全国路線価图的智能查询、批量下载与可视化系统，集成了高性能API服务、Web可视化界面、智能OCR识别与自动化下载工具。适用于地价分析、地理信息研究、批量数据采集等场景。

---

## 目录结构

```
├── app_main.py                # 主入口脚本，启动Web和API服务
├── app/
│   ├── ui/                    # Streamlit前端界面
│   ├── processors/            # 后端处理与API逻辑
│   └── services/              # API服务实现
├── file_system/
│   ├── managers/              # 下载器（rosenka_downloader.py）
│   ├── downloaders/           # URL映射等辅助工具
│   ├── cleaners/              # 数据清理脚本
├── rosenka_data/              # 下载数据与元数据（不建议上传GitHub）
│   ├── metadata/              # 各都道府县元数据
│   └── ...                    # 各地实际PDF等
├── config/                    # 配置与依赖
│   ├── requirements.txt       # 依赖包
│   └── ...
├── venv_simple/               # 虚拟环境（本地使用，不上传）
├── .gitignore                 # Git忽略配置
└── README.md                  # 项目说明文档
```

---

## 主要模块说明

- **主应用（app_main.py）**：一键启动Web界面和API服务，支持多模式运行。
- **Web UI（app/ui/rosenka_web.py）**：Streamlit可视化界面，支持地址检索、OCR识别、智能匹配。
- **API服务（app/services/simple_rosenka_service.py）**：FastAPI接口，供前端和第三方调用。
- **下载器（file_system/managers/rosenka_downloader.py）**：全国路線価图PDF批量下载，自动跳过已存在文件，支持断点续传与并发。
- **数据与元数据（rosenka_data/）**：存放下载PDF和元数据，便于检索和管理。

---

## 安装与环境

1. **克隆项目**
   ```bash
   git clone https://github.com/xihaopark/rosenka_proj.git
   cd rosenka_proj
   ```
2. **创建并激活虚拟环境**
   ```bash
   python3 -m venv venv_simple
   source venv_simple/bin/activate
   ```
3. **安装依赖**
   ```bash
   pip install -r config/requirements.txt
   ```

---

## 启动与用法

### 1. 启动主应用（Web+API）
```bash
python app_main.py
```
- 默认同时启动API服务和Web界面。
- 可通过参数选择模式：
  ```bash
  python app_main.py --mode api   # 仅API
  python app_main.py --mode web   # 仅Web
  python app_main.py --mode both  # 默认，两者都启
  ```

### 2. 批量下载路線価图PDF
```bash
python file_system/managers/rosenka_downloader.py
```
- 自动跳过已下载文件，支持断点续传。
- 下载结果保存在 `rosenka_data/` 目录下。

---

## 数据与同步说明

- **大文件（如PDF、metadata、venv等）不应上传GitHub**，已通过 `.gitignore` 排除。
- 仅同步代码、配置、说明文档等必要内容。
- 如需数据共享，请使用网盘或专用数据同步工具。

---

## GitHub 仓库

- 项目主页：[https://github.com/xihaopark/rosenka_proj](https://github.com/xihaopark/rosenka_proj)
- 如需贡献、提issue或PR，欢迎直接在GitHub操作。

---

## 贡献与开发建议

- 建议使用虚拟环境，避免依赖冲突。
- 代码风格建议遵循PEP8。
- 重要变更请先在本地测试，再推送到远程仓库。
- 如需重置远程仓库内容，可使用：
  ```bash
  git add .
  git commit -m "refactor: 全面重构并替换为新项目结构"
  git push --force origin main
  ```

---

如有问题或建议，欢迎在GitHub仓库留言！ 