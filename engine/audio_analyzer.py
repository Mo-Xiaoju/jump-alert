import librosa
import numpy as np


class AudioAnalyzer:
    """音频爆点检测器：基于多特征融合（RMS + 频谱通量 + 频谱质心 + 频谱平坦度）检测惊吓点。

    通过综合音量、频谱突变、高频成分和噪声特性，提高 Jump Scare 检测准确率。
    """

    def __init__(
        self,
        sr=16000,
        rms_quantile=0.9,
        flux_quantile=0.85,
        centroid_quantile=0.8,
        flatness_quantile=0.7,
        min_gap_sec=0.3,
    ):
        """初始化音频分析器。

        Args:
            sr: 音频采样率。
            rms_quantile: RMS 能量分位数阈值 (0~1)。
            flux_quantile: 频谱通量分位数阈值。
            centroid_quantile: 频谱质心分位数阈值。
            flatness_quantile: 频谱平坦度分位数阈值。
            min_gap_sec: 两个候选事件之间的最小间隔（秒）。
        """
        self.sr = sr
        self.rms_quantile = rms_quantile
        self.flux_quantile = flux_quantile
        self.centroid_quantile = centroid_quantile
        self.flatness_quantile = flatness_quantile
        self.min_gap_sec = min_gap_sec

    def analyze(self, audio_path: str, progress_callback=None) -> list:
        """分析音频文件，返回惊吓点时间列表。

        Args:
            audio_path: 音频文件路径（.wav）。
            progress_callback: 进度回调函数，接收 (percentage, message) 参数。

        Returns:
            惊吓点列表，每个元素为 {"time": 秒数, "intensity": 强度}。
        """
        if progress_callback:
            progress_callback(20, "正在加载音频...")
        y, sr = librosa.load(audio_path, sr=self.sr)

        hop_length = 512
        frame_length = 2048

        if progress_callback:
            progress_callback(25, "正在计算 RMS 能量...")
        # 1. RMS 能量
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        if progress_callback:
            progress_callback(30, "正在计算频谱通量...")
        # 2. 频谱通量（捕捉瞬态冲击）
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        if progress_callback:
            progress_callback(35, "正在计算频谱质心...")
        # 3. 频谱质心（捕捉高频/明亮度）
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length
        )[0]

        if progress_callback:
            progress_callback(40, "正在计算频谱平坦度...")
        # 4. 频谱平坦度（区分噪声类声音和乐音类声音）
        spectral_flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=frame_length, hop_length=hop_length
        )[0]

        # 对齐长度（防止特征长度不一致）
        min_len = min(len(rms), len(spectral_flux), len(spectral_centroid), len(spectral_flatness))
        rms = rms[:min_len]
        spectral_flux = spectral_flux[:min_len]
        spectral_centroid = spectral_centroid[:min_len]
        spectral_flatness = spectral_flatness[:min_len]

        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        if progress_callback:
            progress_callback(45, "正在计算阈值...")
        # 4. 自适应阈值
        rms_th = np.quantile(rms, self.rms_quantile)
        flux_th = np.quantile(spectral_flux, self.flux_quantile)
        centroid_th = np.quantile(spectral_centroid, self.centroid_quantile)
        flatness_th = np.quantile(spectral_flatness, self.flatness_quantile)

        # 5. 组合条件：高冲击 + 高频移 + 够响亮 + 像噪声
        is_sudden = spectral_flux > flux_th
        is_bright = spectral_centroid > centroid_th
        is_loud = rms > rms_th
        is_noise_like = spectral_flatness > flatness_th

        candidate_frames = is_sudden & is_bright & is_loud & is_noise_like
        candidate_idx = np.where(candidate_frames)[0]

        if len(candidate_idx) == 0:
            if progress_callback:
                progress_callback(59, "音频分析完成，未发现候选点")
            return []

        # 6. 合并间隔过近的点，并计算强度
        merged = []
        last_time = -1000
        total_candidates = len(candidate_idx)
        for ci, idx in enumerate(candidate_idx):
            t = times[idx]
            if t - last_time > self.min_gap_sec:
                # 强度计算：综合四个特征的超标程度
                intensity = (
                    (rms[idx] / rms_th if rms_th > 0 else 1)
                    + (spectral_flux[idx] / flux_th if flux_th > 0 else 1)
                    + (spectral_centroid[idx] / centroid_th if centroid_th > 0 else 1)
                    + (spectral_flatness[idx] / flatness_th if flatness_th > 0 else 1)
                ) / 4.0

                merged.append({"time": round(t, 2), "intensity": round(float(intensity), 2)})
                last_time = t

                # 汇报音频分析内部进度（20%~59% 区间）
                if progress_callback and total_candidates > 0:
                    pct = 20 + int((ci / total_candidates) * 39)
                    progress_callback(pct, f"正在合并音频候选点 ({ci+1}/{total_candidates})...")

        if progress_callback:
            progress_callback(59, f"音频分析完成，发现 {len(merged)} 个候选点")
        return merged
