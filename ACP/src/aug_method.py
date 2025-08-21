import torch
import numpy as np
from numpy.fft import ifft  # Added import for ifft

class Augmentation:
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
        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape
        xy_mixed = xy.clone()
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))
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
        num_segments = (total_len + segment_size - 1) // segment_size
        padded_len = num_segments * segment_size
        if padded_len > total_len:
            padding = torch.zeros(
                batch_size, padded_len - total_len, channels, device=xy.device
            )
            xy_padded = torch.cat([xy, padding], dim=1)
        else:
            xy_padded = xy
        segments = xy_padded[:, : num_segments * segment_size, :].view(
            batch_size, num_segments, segment_size, channels
        )
        variances = segments.var(dim=2)
        temporal_weights = torch.linspace(
            0.1, 1.0, steps=segment_size, device=xy.device
        ).view(1, 1, segment_size, 1)
        b_idx = torch.randperm(batch_size)
        for group in channel_groups:
            for c in group:
                low_var_mask = variances[:, :, c] < variance_threshold
                for s in range(num_segments):
                    if low_var_mask[:, s].any():
                        segment_start = s * segment_size
                        segment_end = min(segment_start + segment_size, total_len)
                        segment = xy[:, segment_start:segment_end, c : c + 1]
                        segment_mixed = xy[b_idx, segment_start:segment_end, c : c + 1]
                        mix_weights = (
                            torch.rand(segment.shape, device=xy.device)
                            * mix_rate
                            * temporal_weights[:, :, : segment.shape[1], :]
                        )
                        mask = low_var_mask[:, s].view(batch_size, 1, 1)
                        mix_weights = mix_weights * mask
                        xy_mixed[:, segment_start:segment_end, c : c + 1] = (
                            1 - mix_weights
                        ) * segment + mix_weights * segment_mixed
                        scale = torch.rand(
                            (batch_size, 1, 1), device=xy.device, dtype=torch.float
                        ) * scale_factor * 2 + (1 - scale_factor)
                        scale = scale * mask
                        xy_mixed[:, segment_start:segment_end, c : c + 1] *= scale
        xy_mixed = xy_mixed[:, :total_len, :]
        torch.cuda.empty_cache()
        return xy_mixed

    @staticmethod
    def tf_mix(
        x: torch.Tensor,
        y: torch.Tensor,
        segment_size: int = 24,
        mix_rate: float = 0.15,
        scale_factor: float = 0.05,
        variance_threshold: float = 0.03,
        corr_threshold: float = 0.7,
        freq_mix_rate: float = 0.1,
        dim: int = 1,
    ) -> torch.Tensor:
        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape
        xy_mixed = xy.clone()

        # Compute channel correlations and variances
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))
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

        # Time-domain mixing
        num_segments = (total_len + segment_size - 1) // segment_size
        padded_len = num_segments * segment_size
        if padded_len > total_len:
            padding = torch.zeros(
                batch_size, padded_len - total_len, channels, device=xy.device
            )
            xy_padded = torch.cat([xy, padding], dim=1)
        else:
            xy_padded = xy
        segments = xy_padded[:, : num_segments * segment_size, :].view(
            batch_size, num_segments, segment_size, channels
        )
        variances = segments.var(dim=2)
        b_idx = torch.randperm(batch_size)
        temporal_weights = torch.linspace(
            0.1, 1.0, steps=segment_size, device=xy.device
        ).view(1, 1, segment_size, 1)
        for group in channel_groups:
            for c in group:
                low_var_mask = variances[:, :, c] < variance_threshold
                for s in range(num_segments):
                    if low_var_mask[:, s].any():
                        segment_start = s * segment_size
                        segment_end = min(segment_start + segment_size, total_len)
                        segment = xy[:, segment_start:segment_end, c : c + 1]
                        segment_mixed = xy[b_idx, segment_start:segment_end, c : c + 1]
                        lam = np.random.beta(mix_rate, mix_rate)
                        mix_weights = (
                            torch.ones(segment.shape, device=xy.device)
                            * lam
                            * temporal_weights[:, :, : segment.shape[1], :]
                        )
                        mask = low_var_mask[:, s].view(batch_size, 1, 1)
                        mix_weights = mix_weights * mask
                        xy_mixed[:, segment_start:segment_end, c : c + 1] = (
                            1 - mix_weights
                        ) * segment + mix_weights * segment_mixed

        # Frequency-domain mixing
        xy_f = torch.fft.rfft(xy_mixed, dim=dim)
        amp = torch.abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2  # Skip top 2 frequencies
        m = (torch.rand(xy_f.shape, device=xy_f.device) < freq_mix_rate) & dominant_mask
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f_mixed = torch.complex(freal, fimag)
        xy2 = xy[b_idx]
        xy2_f = torch.fft.rfft(xy2, dim=dim)
        freal2 = xy2_f.real.masked_fill(~m, 0)
        fimag2 = xy2_f.imag.masked_fill(~m, 0)
        xy_f_mixed = torch.complex(freal + freal2, fimag + fimag2)
        xy_mixed = torch.fft.irfft(xy_f_mixed, dim=dim)

        # Apply random scaling
        for group in channel_groups:
            for c in group:
                scale = torch.rand(
                    (batch_size, 1, 1), device=xy.device
                ) * scale_factor * 2 + (1 - scale_factor)
                xy_mixed[:, :, c : c + 1] *= scale

        xy_mixed = xy_mixed[:, :total_len, :]
        torch.cuda.empty_cache()
        return xy_mixed
    @staticmethod
    def vmd_mix(
        x: torch.Tensor,
        y: torch.Tensor,
        K: int = 5,
        alpha: int = 2000,
        tau: float = 0,
        mix_rate: float = 0.15,
        scale_factor: float = 0.05,
        variance_threshold: float = 0.03,
        corr_threshold: float = 0.7,
        dim: int = 1,
    ) -> torch.Tensor:
        """
        VMD-Mix augmentation: Decomposes signals into modes using Variational Mode Decomposition,
        mixes low-variance modes across samples, and reconstructs the signal.
        """
        def vmd(f, alpha, tau, K, DC, init, tol):
            """
            Variational Mode Decomposition (VMD) implementation.
            Input: f (1D signal), alpha (bandwidth constraint), tau (noise tolerance),
                   K (number of modes), DC (include DC component), init (initialization type),
                   tol (convergence tolerance).
            Output: u (time-domain modes), u_hat (frequency-domain modes), omega (center frequencies).
            """
            N = len(f)
            f_ext = np.concatenate((f[::-1], f, f[::-1]))  # Mirror extension
            T = len(f_ext)
            t = np.arange(1, T+1) / T
            if init == 1:
                u_hat = np.zeros((K, T), dtype=complex)
                omega = np.linspace(1, K, K) / (2*N)
            else:
                u_hat = np.random.randn(K, T) + 1j * np.random.randn(K, T)
                omega = np.sort(np.random.rand(K)) / 2
            lambda_hat = np.zeros(T, dtype=complex)
            u_hat_prev = u_hat.copy()
            n = 0
            while True:
                for k in range(K):
                    u_hat[k] = (f_ext - np.sum(u_hat[:k] + u_hat[k+1:], axis=0) + lambda_hat/2) / \
                               (1 + alpha * (omega[k] - np.arange(T)/T)**2)
                for k in range(K):
                    if DC and k == 0:
                        omega[k] = 0
                    else:
                        u_hat_k = u_hat[k]
                        idx = np.where(np.abs(u_hat_k) > tol)[0]
                        if len(idx) > 0:
                            omega[k] = np.sum(t[idx] * np.abs(u_hat_k[idx])**2) / np.sum(np.abs(u_hat_k[idx])**2)
                lambda_hat = lambda_hat + tau * (f_ext - np.sum(u_hat, axis=0))
                max_change = np.max(np.abs(u_hat - u_hat_prev))
                if max_change < tol:
                    break
                u_hat_prev = u_hat.copy()
                n += 1
                if n > 500:
                    break
            u = np.zeros((K, N))
            for k in range(K):
                u[k] = np.real(ifft(u_hat[k][N:2*N]))  # Inverse FFT to get time-domain mode
            return u, np.abs(u_hat[:, N:2*N]), omega

        xy = torch.cat([x, y], dim=dim)
        batch_size, total_len, channels = xy.shape
        xy_mixed = xy.clone()
        corr_matrix = torch.corrcoef(xy[0].permute(1, 0))
        channel_groups = []
        visited = set()
        for c in range(channels):
            if c not in visited:
                group = [c]
                visited.add(c)
                for other_c in range(c + 1, channels):
                    if corr_matrix[c, other_c] > corr_threshold and other_c not in visited:
                        group.append(other_c)
                        visited.add(other_c)
                channel_groups.append(group)
        b_idx = torch.randperm(batch_size)
        for group in channel_groups:
            for c in group:
                xy_c = xy[:, :, c].cpu().numpy()  # Move to CPU for VMD
                for b in range(batch_size):
                    u, _, omega = vmd(xy_c[b], alpha, tau, K, DC=False, init=1, tol=1e-7)
                    u_var = np.var(u, axis=1)
                    low_var_mask = u_var < variance_threshold
                    if np.any(low_var_mask):
                        u_mixed = u.copy()
                        u_other = vmd(xy[b_idx[b], :, c].cpu().numpy(), alpha, tau, K, DC=False, init=1, tol=1e-7)[0]
                        lam = np.random.beta(mix_rate, mix_rate)
                        u_mixed[low_var_mask] = (1 - lam) * u[low_var_mask] + lam * u_other[low_var_mask]
                        xy_mixed[b, :, c] = torch.from_numpy(np.sum(u_mixed, axis=0)).to(torch.device).float()  # type: ignore
                scale = torch.rand((batch_size, 1), device=xy.device) * scale_factor * 2 + (1 - scale_factor)
                xy_mixed[:, :, c] *= scale.unsqueeze(1)
        torch.cuda.empty_cache()
        return xy_mixed
