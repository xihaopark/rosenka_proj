#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_main.py
路線価图应用主脚本
"""

import sys
from pathlib import Path

# 添加app路径
sys.path.append(str(Path(__file__).parent / "app" / "ui"))

from simple_local_app import main

if __name__ == "__main__":
    main()
