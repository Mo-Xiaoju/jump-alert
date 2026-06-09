import cv2
import numpy as np


class VideoAnalyzer:
    """视频帧差检测器：基于相邻帧差异检测视觉惊吓点。

    计算相邻帧的像素差异总和，当突变超过阈值时标记为候选点。
    """

    def __init__(self, threshold_multiplier=3.0, target_width=360):
        """初始化视频分析器。

        Args:
            threshold_multiplier: 阈值倍数，局部平均帧差的 N 倍视为突变。
            target_width: 分析时缩放到的目标宽度，降低计算量。
        """
        self.threshold_multiplier = threshold_multiplier
        self.target_width = target_width

    def analyze(self, video_path: str) -> list:
        """分析视频文件，返回惊吓点时间列表。

        Args:
            video_path: 视频文件路径。

        Returns:
            惊吓点列表，每个元素为 {"time": 秒数, "intensity": 强度}。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        diffs = []
        prev_frame = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.target_width, int(frame.shape[0] * self.target_width / frame.shape[1])))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                diff = np.sum(cv2.absdiff(gray, prev_frame))
                diffs.append(diff)

            prev_frame = gray
            frame_count += 1

        cap.release()

        if not diffs:
            return []

        diffs = np.array(diffs)
        window = max(10, len(diffs) // 100)
        candidates = []

        for i in range(window, len(diffs) - window):
            local_mean = np.mean(diffs[i - window:i + window])
            if diffs[i] > local_mean * self.threshold_multiplier:
                time_sec = (i + 1) / fps
                candidates.append({
                    "time": round(time_sec, 2),
                    "intensity": round(float(diffs[i] / local_mean), 2)
                })

        merged = self._merge_close_points(candidates, min_gap=1.0)
        return merged

    @staticmethod
    def _merge_close_points(points: list, min_gap: float = 1.0) -> list:
        """合并时间相近的候选点，保留强度最高的一个。"""
        if not points:
            return []
        merged = [points[0]]
        for p in points[1:]:
            if p["time"] - merged[-1]["time"] < min_gap:
                if p["intensity"] > merged[-1]["intensity"]:
                    merged[-1] = p
            else:
                merged.append(p)
        return merged
