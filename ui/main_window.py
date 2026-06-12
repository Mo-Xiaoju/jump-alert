import os
import cv2
import json
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow, QListWidget, QListWidgetItem, QMessageBox, QFileDialog
from PySide6.QtCore import QTimer, Qt, QUrl, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QDropEvent, QDragEnterEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import sys
from PySide6.QtWidgets import QApplication
UI_PATH = os.path.join(os.path.dirname(__file__), "main_window.ui")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}

from pydub import AudioSegment

def setup_ffmpeg():
    if getattr(sys, 'frozen', False):
        # 打包后运行：ffmpeg 在临时解压目录 sys._MEIPASS 中
        ffmpeg_path = os.path.join(sys._MEIPASS, "ffmpeg.exe")
    else:
        # 开发环境运行：ffmpeg 在项目根目录（当前文件在 ui/ 下，需向上一级）
        ffmpeg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ffmpeg.exe")
    
    if os.path.exists(ffmpeg_path):
        AudioSegment.converter = ffmpeg_path
    else:
        print(f"警告：未找到 FFmpeg ({ffmpeg_path})，音频处理功能可能不可用。")

setup_ffmpeg()

class AnalysisWorker(QThread):
    """后台分析线程：执行 Detector 检测流程，避免阻塞 UI。"""
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int, str)  # (百分比, 状态消息)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from engine.detector import Detector

            # 定义进度回调函数，将进度转发给 UI 线程
            def on_progress(pct, msg):
                self.progress.emit(pct, msg)

            detector = Detector()
            results = detector.detect(self.video_path, progress_callback=on_progress)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class DropListWidget(QListWidget):
    """自定义列表控件，支持拖拽视频文件到列表中。

    继承自 QListWidget，重写拖拽事件以实现文件拖入功能。
    仅接受 VIDEO_EXTENSIONS 中定义的视频格式文件。
    """

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件：判断拖入的文件是否为视频格式，是则接受。"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """拖拽放下事件：将拖入的视频文件路径添加到列表中。"""
        event.acceptProposedAction()
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
                self.addItem(QListWidgetItem(path))


class MainWindow(QMainWindow):
    """主窗口类：负责视频播放、文件管理、分析控制等核心交互逻辑。

    使用 QUiLoader 动态加载 .ui 文件构建界面。
    视频帧由 OpenCV 读取并显示到 QLabel，音频由 QMediaPlayer 独立播放。
    通过监听音频播放位置实现音画同步。
    """

    def __init__(self):
        """初始化主窗口：加载 UI、初始化播放器、绑定信号。"""
        super().__init__()
        loader = QUiLoader()
        loader.registerCustomWidget(DropListWidget)
        self.ui = loader.load(UI_PATH, self)
        self.setCentralWidget(self.ui.centralwidget)

        self.cap = None
        self.timer = QTimer(self)
        self.current_video = None
        self.is_playing = False
        self.fps = 30
        self.frame_interval = 0
        

        self.audio_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_player.positionChanged.connect(self._on_position_changed)

        self._connect_signals()

    def _connect_signals(self):
        """绑定 UI 控件的信号与槽函数。"""
        self.ui.fileListWidget.itemDoubleClicked.connect(self._on_file_double_clicked)
        self.ui.playButton.clicked.connect(self._play)
        self.ui.pauseButton.clicked.connect(self._pause)
        self.ui.stopButton.clicked.connect(self._stop)
        self.ui.processButton.clicked.connect(self._process)
        self.ui.selectFileButton.clicked.connect(self._select_file)
        self.timer.timeout.connect(self._read_frame)

    def _select_file(self):
        """弹出文件选择对话框，将选中的视频文件添加到列表。"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.webm);;所有文件 (*)"
        )
        for path in files:
            existing = [self.ui.fileListWidget.item(i).text() for i in range(self.ui.fileListWidget.count())]
            if path not in existing:
                self.ui.fileListWidget.addItem(QListWidgetItem(path))

    def _on_file_double_clicked(self, item):
        """双击列表项：加载并准备播放该视频。"""
        self._load_video(item.text())

    def _load_video(self, path):
        """加载视频文件：初始化 OpenCV 捕获器和音频播放器。

        读取视频 FPS 用于计算帧间隔，设置音频源，显示第一帧。
        """
        self._stop()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频文件:\n{path}")
            return
        self.current_video = path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        self.frame_interval = 1000 / self.fps
        self.audio_player.setSource(QUrl.fromLocalFile(path))
        self.ui.progressSlider.setValue(0)
        self._read_frame()

    def _on_position_changed(self, position):
        """音频位置变化回调：根据音频播放进度同步视频帧位置。

        当视频帧与音频位置偏差超过 2 帧时，跳转视频到正确位置。
        """
        if not self.is_playing:
            return
        target_frame = int(position / 1000 * self.fps)
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if abs(target_frame - current_frame) > 2:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def _play(self):
        """播放：启动帧读取定时器，开始音频播放。"""
        if not self.cap or not self.current_video:
            return
        self.is_playing = True
        self.timer.start(int(self.frame_interval))
        self.audio_player.play()

    def _pause(self):
        """暂停：停止帧读取定时器，暂停音频播放。"""
        self.is_playing = False
        self.timer.stop()
        self.audio_player.pause()

    def _stop(self):
        """停止：释放视频资源，重置 UI 状态。"""
        self.is_playing = False
        self.timer.stop()
        self.audio_player.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_video = None
        self.ui.videoLabel.setText("拖入视频文件到左侧，点击播放")
        self.ui.progressSlider.setValue(0)
        self.ui.timeLabel.setText("00:00 / 00:00")

    def _read_frame(self):
        """读取并显示下一帧视频。

        从 OpenCV 捕获器读取一帧，转换为 QImage 显示到 QLabel。
        更新进度条和时间标签。播放结束时自动循环到开头。
        """
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if total_frames > 0:
            self.ui.progressSlider.setMaximum(total_frames)
            self.ui.progressSlider.setValue(current_frame)

        current_sec = current_frame / self.fps if self.fps > 0 else 0
        total_sec = total_frames / self.fps if self.fps > 0 else 0
        self.ui.timeLabel.setText(f"{self._format_time(current_sec)} / {self._format_time(total_sec)}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.ui.videoLabel.width(),
            self.ui.videoLabel.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.ui.videoLabel.setPixmap(scaled)

    def _process(self):
        """处理按钮点击：启动后台线程执行视频分析。

        将视频分离为音频和视频，分别传给 AudioAnalyzer 和 VideoAnalyzer 处理。
        分析完成后生成 jump_scares.json 文件。
        """
        if not self.current_video:
            QMessageBox.information(self, "提示", "请先选择一个视频文件")
            return

        self.ui.processButton.setEnabled(False)
        self.ui.processButton.setText("分析中...")

        self.worker = AnalysisWorker(self.current_video)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.progress.connect(self._on_analysis_progress)
        self.worker.start()

    def _on_analysis_progress(self, value, message):
        """分析进度回调：更新进度条和按钮文字。"""
        self.ui.progressBar.setValue(value)
        self.ui.processButton.setText(message)

    def _on_analysis_finished(self, results):
        """分析完成回调：显示结果并保存 JSON。"""
        self.ui.processButton.setEnabled(True)
        self.ui.processButton.setText("开始处理")
        self.ui.progressBar.setValue(0)

        output_path = os.path.splitext(self.current_video)[0] + "_jump_scares.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        QMessageBox.information(
            self, "分析完成",
            f"发现 {len(results)} 个惊吓点\n结果已保存到:\n{output_path}"
        )

    def _on_analysis_error(self, error_msg):
        """分析错误回调：显示错误信息。"""
        self.ui.processButton.setEnabled(True)
        self.ui.processButton.setText("开始处理")
        self.ui.progressBar.setValue(0)
        QMessageBox.critical(self, "分析失败", f"分析过程中出错:\n{error_msg}")

    @staticmethod
    def _format_time(seconds):
        """将秒数格式化为 MM:SS 字符串。"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
