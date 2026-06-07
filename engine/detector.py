from .audio_analyzer import AudioAnalyzer
from .video_analyzer import VideoAnalyzer


class Detector:
    def __init__(self):
        self.audio_analyzer = AudioAnalyzer()
        self.video_analyzer = VideoAnalyzer()

    def detect(self, media_path: str):
        raise NotImplementedError
