import librosa
import numpy as np


class AudioAnalyzer:
    """音频爆点检测器：基于多特征融合（RMS + 频谱通量 + 频谱质心）检测惊吓点。

    通过综合音量、频谱突变和高频成分，提高 Jump Scare 检测准确率。
    """

    def __init__(
        self,
        sr=16000,
        rms_quantile=0.9,
        flux_quantile=0.85,
        centroid_quantile=0.8,
        min_gap_sec=0.3,
    ):
        """初始化音频分析器。

        Args:
            sr: 音频采样率。
            rms_quantile: RMS 能量分位数阈值 (0~1)。
            flux_quantile: 频谱通量分位数阈值。
            centroid_quantile: 频谱质心分位数阈值。
            min_gap_sec: 两个候选事件之间的最小间隔（秒）。
        """
        self.sr = sr
        self.rms_quantile = rms_quantile
        self.flux_quantile = flux_quantile
        self.centroid_quantile = centroid_quantile
        self.min_gap_sec = min_gap_sec

    def analyze(self, audio_path: str) -> list:
        """分析音频文件，返回惊吓点时间列表。

        Args:
            audio_path: 音频文件路径（.wav）。

        Returns:
            惊吓点列表，每个元素为 {"time": 秒数, "intensity": 强度}。
        """
        y, sr = librosa.load(audio_path, sr=self.sr)

        hop_length = 512
        frame_length = 2048

        # 1. RMS 能量
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # 2. 频谱通量（捕捉瞬态冲击）
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # 3. 频谱质心（捕捉高频/明亮度）
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
        )[0]

        # 对齐长度（防止特征长度不一致）
        min_len = min(len(rms), len(spectral_flux), len(spectral_centroid))
        rms = rms[:min_len]
        spectral_flux = spectral_flux[:min_len]
        spectral_centroid = spectral_centroid[:min_len]

        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        # 4. 自适应阈值
        rms_th = np.quantile(rms, self.rms_quantile)
        flux_th = np.quantile(spectral_flux, self.flux_quantile)
        centroid_th = np.quantile(spectral_centroid, self.centroid_quantile)

        # 5. 组合条件：高冲击 + 高频移 + 够响亮
        is_sudden = spectral_flux > flux_th
        is_bright = spectral_centroid > centroid_th
        is_loud = rms > rms_th

        candidate_frames = is_sudden & is_bright & is_loud
        candidate_idx = np.where(candidate_frames)[0]

        if len(candidate_idx) == 0:
            return []

        # 6. 合并间隔过近的点，并计算强度
        merged = []
        last_time = -1000
        for idx in candidate_idx:
            t = times[idx]
            if t - last_time > self.min_gap_sec:
                # 强度计算：综合三个特征的超标程度
                intensity = (
                    (rms[idx] / rms_th if rms_th > 0 else 1)
                    + (spectral_flux[idx] / flux_th if flux_th > 0 else 1)
                    + (spectral_centroid[idx] / centroid_th if centroid_th > 0 else 1)
                ) / 3.0

                merged.append({"time": round(t, 2), "intensity": round(float(intensity), 2)})
                last_time = t

        return merged
