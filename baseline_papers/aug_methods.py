import torch
import numpy as np
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from PyEMD import EMD


# Augmentation class
class Augmentation:
    @staticmethod
    def freq_mask(
        x: torch.Tensor, y: torch.Tensor, rate: float = 0.5, dim: int = 1
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        m = torch.rand(xy_f.shape, device=xy_f.device) < rate
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)
        return xy

    @staticmethod
    def freq_mix(
        x: torch.Tensor, y: torch.Tensor, rate: float = 0.5, dim: int = 1
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        m = torch.rand(xy_f.shape, device=xy_f.device) < rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m, dominant_mask)
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        b_idx = np.arange(x.shape[0])
        np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = torch.cat([x2, y2], dim=dim)
        xy2_f = torch.fft.rfft(xy2, dim=dim)
        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m, 0)
        fimag2 = xy2_f.imag.masked_fill(m, 0)
        freal += freal2
        fimag += fimag2
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)
        return xy

    @staticmethod
    def wave_mask(
        x: torch.Tensor,
        y: torch.Tensor,
        rates: list,
        wavelet: str = "db1",
        level: int = 2,
        dim: int = 1,
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        rate_tensor = torch.tensor(
            rates[: level + 1], device=x.device
        )  # Ensure rates match level
        xy = xy.permute(0, 2, 1).to(x.device).to(torch.float64)
        try:
            dwt = (
                DWT1DForward(J=level, wave=wavelet, mode="symmetric")
                .to(x.device)
                .double()
            )
        except:
            print(f"Wavelet {wavelet} not supported, falling back to db1")
            dwt = (
                DWT1DForward(J=level, wave="db1", mode="symmetric")
                .to(x.device)
                .double()
            )
        cA, cDs = dwt(xy)
        mask = torch.rand_like(cA).to(cA.device) < rate_tensor[0]
        cA = cA.masked_fill(mask, 0)
        masked_cDs = []
        for i, cD in enumerate(cDs, 1):
            mask_cD = torch.rand_like(cD).to(cD.device) < rate_tensor[i]
            cD = cD.masked_fill(mask_cD, 0)
            masked_cDs.append(cD)
        try:
            idwt = DWT1DInverse(wave=wavelet, mode="symmetric").to(x.device).double()
        except:
            idwt = DWT1DInverse(wave="db1", mode="symmetric").to(x.device).double()
        reconstructed = idwt((cA, masked_cDs))
        reconstructed = reconstructed.permute(0, 2, 1)
        return reconstructed.float()

    @staticmethod
    def wave_mix(
        x: torch.Tensor,
        y: torch.Tensor,
        rates: list,
        wavelet: str = "db1",
        level: int = 2,
        dim: int = 1,
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        batch_size, _, _ = xy.shape
        rate_tensor = torch.tensor(rates[: level + 1], device=x.device)
        xy = xy.permute(0, 2, 1).to(x.device).to(torch.float64)
        b_idx = torch.randperm(batch_size)
        xy2 = xy[b_idx].to(torch.float64)
        try:
            dwt = (
                DWT1DForward(J=level, wave=wavelet, mode="symmetric")
                .to(x.device)
                .double()
            )
        except:
            print(f"Wavelet {wavelet} not supported, falling back to db1")
            dwt = (
                DWT1DForward(J=level, wave="db1", mode="symmetric")
                .to(x.device)
                .double()
            )
        cA1, cDs1 = dwt(xy)
        cA2, cDs2 = dwt(xy2)
        mask = torch.rand_like(cA1).to(cA1.device) < rate_tensor[0]
        cA_mixed = cA1.masked_fill(mask, 0) + cA2.masked_fill(~mask, 0)
        mixed_cDs = []
        for i, (cD1, cD2) in enumerate(zip(cDs1, cDs2), 1):
            mask = torch.rand_like(cD1).to(cD1.device) < rate_tensor[i]
            cD_mixed = cD1.masked_fill(mask, 0) + cD2.masked_fill(~mask, 0)
            mixed_cDs.append(cD_mixed)
        try:
            idwt = DWT1DInverse(wave=wavelet, mode="symmetric").to(x.device).double()
        except:
            idwt = DWT1DInverse(wave="db1", mode="symmetric").to(x.device).double()
        reconstructed = idwt((cA_mixed, mixed_cDs))
        reconstructed = reconstructed.permute(0, 2, 1)
        return reconstructed.float()

    @staticmethod
    def emd_aug(x: torch.Tensor) -> torch.Tensor:
        b, n_imf, t, c = x.size()
        inp = x.permute(0, 2, 1, 3).reshape(b, t, n_imf * c)
        if torch.rand(1) >= 0.5:
            w = 2 * torch.rand((b, 1, n_imf * c)).to(x.device)
        else:
            w = torch.ones((b, 1, n_imf * c)).to(x.device)
        w_exp = w.expand(-1, t, -1)
        out = w_exp * inp
        out = out.reshape(b, t, n_imf, c).sum(dim=2)
        return out

    @staticmethod
    def mix_aug(
        batch_x: torch.Tensor, batch_y: torch.Tensor, lambd: float = 0.5
    ) -> tuple:
        inds2 = torch.randperm(batch_x.shape[0])
        lam = np.random.beta(lambd, lambd)
        batch_x = lam * batch_x[inds2] + (1 - lam) * batch_x
        batch_y = lam * batch_y[inds2] + (1 - lam) * batch_y
        return batch_x, batch_y

    @staticmethod
    def adaptive_channel_preserve(
        x: torch.Tensor,
        y: torch.Tensor,
        segment_size: int = 24,
        mix_rate: float = 0.15,
        scale_factor: float = 0.05,
        variance_threshold: float = 0.03,
        corr_threshold: float = 0.7,
        dim: int = 1,
    ) -> torch.Tensor:
        """
        Adaptive-Channel-Preserve: Mixes low-variance segments within correlated channel groups, preserves high-variance segments,
        and applies subtle scaling to low-variance segments.

        Args:
            x: Input tensor (batch_size, seq_len, channels)
            y: Target tensor (batch_size, pred_len, channels)
            segment_size: Size of each segment for variance analysis
            mix_rate: Maximum mixing weight for low-variance segments
            scale_factor: Scaling range for low-variance segments (e.g., [1-scale_factor, scale_factor + 1])
            variance_threshold: Threshold to identify low-variance segments
            corr_threshold: Correlation threshold for channel grouping
            dim: Time dimension for concatenation

        Returns:
            Augmented tensor (batch_size, seq_len + pred_len, channels)
        """
        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape

        # Initialize output tensor
        xy_mixed = xy.clone()

        # Compute channel correlations (using first batch sample for consistency)
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))  # Shape: (channels, channels)

        # Group channels based on correlation
        channel_groups = []
        visited = set()
        for c in range(channels):
            if c not in visited:
                group = [c]
                visited.add(c)
                for other_c in range(c + 1, channels):
                    if (
                        corr_matrix[c, other_c] > corr_threshold
                        and other_c not in visited
                    ):
                        group.append(other_c)
                        visited.add(other_c)
                channel_groups.append(group)

        # Divide sequence into non-overlapping segments
        num_segments = (
            total_len + segment_size - 1
        ) // segment_size  # Ceiling division
        padded_len = num_segments * segment_size
        if padded_len > total_len:
            # Pad xy to match padded_len
            padding = torch.zeros(
                batch_size, padded_len - total_len, channels, device=xy.device
            )
            xy_padded = torch.cat([xy, padding], dim=1)
        else:
            xy_padded = xy

        # Reshape into segments
        segments = xy_padded[:, : num_segments * segment_size, :].view(
            batch_size, num_segments, segment_size, channels
        )

        # Compute variance for each segment
        variances = segments.var(dim=2)  # Shape: (batch_size, num_segments, channels)

        # Temporal weights for mixing
        temporal_weights = torch.linspace(
            0.1, 1.0, steps=segment_size, device=xy.device
        ).view(1, 1, segment_size, 1)

        # Process each channel group
        b_idx = torch.randperm(batch_size)
        for group in channel_groups:
            for c in group:
                low_var_mask = (
                    variances[:, :, c] < variance_threshold
                )  # Shape: (batch_size, num_segments)
                for s in range(num_segments):
                    if low_var_mask[:, s].any():
                        # Extract segment
                        segment_start = s * segment_size
                        segment_end = min(segment_start + segment_size, total_len)
                        segment = xy[
                            :, segment_start:segment_end, c : c + 1
                        ]  # Shape: (batch_size, segment_size, 1)
                        segment_mixed = xy[
                            b_idx, segment_start:segment_end, c : c + 1
                        ]  # Shape: (batch_size, segment_size, 1)

                        # Apply mixing weights
                        mix_weights = (
                            torch.rand(
                                segment.shape, device=xy.device
                            )  # Shape: (batch_size, segment_size, 1)
                            * mix_rate
                            * temporal_weights[
                                :, :, : segment.shape[1], :
                            ]  # Shape: (1, 1, segment_size, 1)
                        )
                        # Apply low-variance mask correctly
                        mask = low_var_mask[:, s].view(
                            batch_size, 1, 1
                        )  # Shape: (batch_size, 1, 1)
                        mix_weights = (
                            mix_weights * mask
                        )  # Shape: (batch_size, segment_size, 1)

                        xy_mixed[:, segment_start:segment_end, c : c + 1] = (
                            1 - mix_weights
                        ) * segment + mix_weights * segment_mixed

                        # Apply scaling to low-variance segments
                        scale = torch.rand(
                            (batch_size, 1, 1), device=xy.device, dtype=torch.float
                        ) * scale_factor * 2 + (
                            1 - scale_factor
                        )  # Shape: (batch_size, 1, 1)
                        scale = scale * mask  # Shape: (batch_size, 1, 1)
                        xy_mixed[:, segment_start:segment_end, c : c + 1] *= scale

        return xy_mixed[:, :total_len, :]  # Trim padding if added
