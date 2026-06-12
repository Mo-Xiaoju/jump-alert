# JumpScareWarner

> 恐怖视频高能预警系统 — 自动检测视频中的惊吓点（Jump Scare），提前预警，让观影不再心跳加速。

## 功能特性

- **自动化分析**：基于音频爆点检测 + 视频帧差分析，无需人工标注即可生成惊吓标记
- **多特征融合**：综合 RMS 等特征，精准识别惊吓点
- **可视化界面**：PySide6 构建的现代化 GUI，支持拖拽导入、视频预览、进度实时反馈
- **标准输出**：生成 `jump_scares.json` 标记文件，可用于本地播放器预警或浏览器扩展联动
- **零依赖分发**：PyInstaller 打包为单文件 EXE，用户无需安装 Python 即可运行


## 快速开始

### 开发环境

```bash
# 1. 克隆项目
git clone https://github.com/your-username/JumpScareWarner.git
cd JumpScareWarner

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载 FFmpeg
# 将 ffmpeg.exe 放入项目根目录（与 main.py 同级）
# 下载地址：https://ffmpeg.org/download.html

# 4. 运行程序
python main.py
```

### 打包为 EXE

```bash
# 确保 ffmpeg.exe 在项目根目录
pyinstaller --noconfirm --onefile --windowed --name "JumpScareWarner" \
  --add-binary "ffmpeg.exe;." \
  --add-data "ui;ui" \
  --add-data "models;models" \
  --hidden-import PySide6 \
  main.py

# 产物在 dist/JumpScareWarner.exe
```

## 使用说明

1. **导入视频**：拖拽视频文件到左侧列表，或点击"选择视频文件"按钮
2. **预览播放**：双击列表中的文件，使用内置播放器预览
3. **开始分析**：点击"开始处理"按钮，等待分析完成
4. **查看结果**：分析完成后自动生成 `xxx_jump_scares.json` 文件

### 输出格式

```json
[
  {
    "time": 1234.5,
    "type": "jumpscare",
    "intensity": "high",
    "audio_intensity": 2.35,
    "video_intensity": 1.87
  },
  {
    "time": 2256.8,
    "type": "audio_spike",
    "intensity": "medium",
    "audio_intensity": 1.92
  }
]
```

## 项目结构

```
JumpScareWarner/
── main.py                  # 程序入口
├── ui/                      # 界面文件
│   ├── main_window.ui       # Qt Designer 界面设计
│   └── main_window.py       # 界面逻辑
├── engine/                  # 核心分析引擎
│   ├── audio_analyzer.py    # 音频爆点检测（四维特征融合）
│   ├── video_analyzer.py    # 视频帧差检测
│   ── detector.py          # 检测调度器
── models/                  # AI 模型存放目录（预留）
├── utils/                   # 通用工具函数（预留）
── requirements.txt         # 项目依赖
├── .gitignore              # Git 忽略规则
└── README.md               # 项目说明
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 界面 | PySide6 (Qt) |
| 视频分析 | OpenCV-Python |
| 音频分析 | librosa |
| 打包 | PyInstaller |
| 音频处理 | FFmpeg |

## 注意事项

- **FFmpeg 依赖**：程序需要 `ffmpeg.exe` 才能进行音频提取。开发时放在项目根目录，打包后自动内置。



