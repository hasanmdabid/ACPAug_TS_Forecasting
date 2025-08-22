import torch
import numpy as np
import pywt
from numpy.fft import fft, ifft


class Augmentation:
    def __init__(self):
        pass  # No initialization needed for CPU-based operations

    def wave_freq_aug(
        self, x, y, mask_rate=0.3, wavelet="db2", level=3, lambd=None, dim=1
    ):
        """
        Apply Wavelet-Frequency Augmentation (WaveFreqAug).
        Decomposes time series using DWT, applies frequency masking to detail coefficients,
        reconstructs with IDWT, and mixes with original series.
        Args:
            x: Input tensor (batch_size, seq_len, enc_in)
            y: Ground truth (batch_size, pred_len, enc_in)
            mask_rate: Fraction of frequencies to mask in detail coefficients
            wavelet: Wavelet type (e.g., 'db2')
            level: Decomposition level (default 3)
            lambd: Mixing coefficient (if None, sampled from Beta(0.5, 0.5))
            dim: Dimension to apply augmentation (typically 1 for time axis)
        Returns:
            Augmented tensor (batch_size, seq_len + pred_len, enc_in)
        """
        # Combine input and ground truth
        xy = torch.cat([x, y], dim=1)  # Shape: (batch_size, seq_len + pred_len, enc_in)
        batch_size, total_len, enc_in = xy.shape
        xy_np = xy.cpu().numpy()  # Move to CPU for wavelet and FFT operations
        xy_aug = np.zeros_like(xy_np)

        # Sample mixing coefficient if not provided
        if lambd is None:
            lambd = np.random.beta(0.5, 0.5)

        # Process each sample and channel
        for b in range(batch_size):
            for c in range(enc_in):
                # Wavelet decomposition
                coeffs = pywt.wavedec(
                    xy_np[b, :, c], wavelet=wavelet, level=level, axis=0
                )
                approx = coeffs[0]  # Approximation coefficients
                details = coeffs[1:]  # Detail coefficients

                # Frequency augmentation on detail coefficients
                for i in range(len(details)):
                    freq = fft(details[i])
                    mask = np.random.rand(len(freq)) < mask_rate
                    freq[mask] = 0  # Mask frequencies
                    details[i] = np.real(ifft(freq))

                # Reconstruct augmented series
                max_level = min(level, pywt.dwt_max_level(len(xy_np[b, :, c]), wavelet))
                if max_level < level:
                    coeffs = coeffs[: max_level + 1] + [None] * (level - max_level)
                xy_aug[b, :, c] = pywt.waverec(
                    [approx] + details, wavelet=wavelet, axis=0
                )[:total_len]

        # Mix with original
        xy_aug = lambd * xy_aug + (1 - lambd) * xy_np
        return torch.tensor(xy_aug, dtype=torch.float32)
