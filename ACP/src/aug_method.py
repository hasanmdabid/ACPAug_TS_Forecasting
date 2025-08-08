import torch
import torch.nn as nn
import numpy as np
# Augmentation class
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
