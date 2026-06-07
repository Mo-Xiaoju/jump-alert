import os
import cv2
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow, QListWidget, QListWidgetItem, QMessageBox, QFileDialog
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QImage, QPixmap, QDropEvent, QDragEnterEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import sys
from PySide6.QtWidgets import QApplication

UI_PATH = os.path.join(os.path.dirname(__file__), "main_window.ui")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}


class DropListWidget(QListWidget):
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        event.acceptProposedAction()
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
                self.addItem(QListWidgetItem(path))


class MainWindow(QMainWindow):
    def __init__(self):
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
        self.last_frame_time = 0

        self.audio_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_player.positionChanged.connect(self._on_position_changed)

        self._connect_signals()

    def _connect_signals(self):
        self.ui.fileListWidget.itemDoubleClicked.connect(self._on_file_double_clicked)
        self.ui.playButton.clicked.connect(self._play)
        self.ui.pauseButton.clicked.connect(self._pause)
        self.ui.stopButton.clicked.connect(self._stop)
        self.ui.processButton.clicked.connect(self._process)
        self.ui.selectFileButton.clicked.connect(self._select_file)
        self.timer.timeout.connect(self._read_frame)

    def _select_file(self):
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
        self._load_video(item.text())

    def _load_video(self, path):
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
        if not self.is_playing:
            return
        target_frame = int(position / 1000 * self.fps)
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if abs(target_frame - current_frame) > 2:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def _play(self):
        if not self.cap or not self.current_video:
            return
        self.is_playing = True
        self.timer.start(int(self.frame_interval))
        self.audio_player.play()

    def _pause(self):
        self.is_playing = False
        self.timer.stop()
        self.audio_player.pause()

    def _stop(self):
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
        if not self.current_video:
            QMessageBox.information(self, "提示", "请先选择一个视频文件")
            return
        QMessageBox.information(self, "处理中", f"开始处理视频:\n{self.current_video}")

    @staticmethod
    def _format_time(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
