import os
import json
from pydub import AudioSegment
from .audio_analyzer import AudioAnalyzer
from .video_analyzer import VideoAnalyzer


class Detector:
    """检测调度器：负责音视频分离、调用分析器、合并结果。

    工作流程：
    1. 从视频文件中提取音频为 
    2. 将音频传给 AudioAnalyzer 检测爆点
    3. 将视频传给 VideoAnalyzer 检测帧差突变
    4. 合并两个分析器的结果，生成 jump_scares.json
    """

    def __init__(self, output_dir: str = None):
        """初始化检测器。

        Args:
            output_dir: 临时文件和结果输出目录，默认为视频同目录。
        """
        self.audio_analyzer = AudioAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        self.output_dir = output_dir

    def detect(self, media_path: str, output_path: str = None, progress_callback=None) -> list:
        """执行完整检测流程。

        Args:
            media_path: 视频文件路径。
            output_path: 结果 JSON 输出路径，默认生成 jump_scares.json。
            progress_callback: 进度回调函数，接收 (percentage, message) 参数。

        Returns:
            惊吓点列表。
        """
        if output_path is None:
            base = os.path.splitext(media_path)[0]
            output_path = base + "_jump_scares.json"

        if self.output_dir is None:
            self.output_dir = os.path.dirname(media_path)

        audio_path = os.path.join(self.output_dir, "_temp_audio.wav")

        try:
            if progress_callback:
                progress_callback(10, "正在提取音频...")
            self._extract_audio(media_path, audio_path)
            
            if progress_callback:
                progress_callback(20, "正在分析音频爆点...")
            audio_results = self.audio_analyzer.analyze(audio_path, progress_callback)
            
            if progress_callback:
                progress_callback(60, "正在分析视频帧差...")
            video_results = self.video_analyzer.analyze(media_path, progress_callback)
            
            if progress_callback:
                progress_callback(90, "正在合并结果...")
            merged = self._merge_results(audio_results, video_results)

            if progress_callback:
                progress_callback(95, "正在保存结果...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

            if progress_callback:
                progress_callback(100, "分析完成")
            return merged
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _extract_audio(self, video_path: str, audio_path: str):
        """从视频文件中提取音频为 WAV 格式。

        Args:
            video_path: 视频文件路径。
            audio_path: 输出音频文件路径。
        """
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(audio_path, format="wav")

    @staticmethod
    def _merge_results(audio_results: list, video_results: list, time_window: float = 1.0) -> list:
        """合并音频和视频分析结果。

        规则：
        - 音频和视频候选点时间接近（< time_window）则合并为高置信度惊吓点
        - 单独出现的标记为中等置信度

        Args:
            audio_results: 音频分析结果列表。
            video_results: 视频分析结果列表。
            time_window: 合并时间窗口（秒）。

        Returns:
            合并后的惊吓点列表，按时间排序。
        """
        merged = []
        used_video = set()

        for a in audio_results:
            matched = False
            for vi, v in enumerate(video_results):
                if vi in used_video:
                    continue
                if abs(a["time"] - v["time"]) < time_window:
                    merged.append({
                        "time": round((a["time"] + v["time"]) / 2, 2),
                        "type": "jumpscare",
                        "intensity": "high",
                        "audio_intensity": a["intensity"],
                        "video_intensity": v["intensity"]
                    })
                    used_video.add(vi)
                    matched = True
                    break
            if not matched:
                merged.append({
                    "time": a["time"],
                    "type": "audio_spike",
                    "intensity": "medium",
                    "audio_intensity": a["intensity"]
                })

        for vi, v in enumerate(video_results):
            if vi not in used_video:
                merged.append({
                    "time": v["time"],
                    "type": "visual_spike",
                    "intensity": "medium",
                    "video_intensity": v["intensity"]
                })

        merged.sort(key=lambda x: x["time"])
        return merged
