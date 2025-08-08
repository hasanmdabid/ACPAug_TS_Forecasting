# Model Parameters
epochs = 30
learning_rate = 0.01
patience = 10
num_iterations = 5
label_len = 0

# Dataset configurations
dataset_configs = {
    "ETTh1": {
        "data_path": "./dataset/ETTh1.csv",
        "data_name": "ETTh1",
        "seq_len": 336,
        "enc_in": 7,
        "batch_size": 64,
        "pred_lens": [96, 192, 336, 720],
        "aug_types": ["Adaptive-Channel-Preserve"],
        "aug_params": {
            96: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            192: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            336: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 36,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
                },
            },
            720: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 48,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
                },
            },
        },
    },
    "ETTh2": {
        "data_path": "./dataset/ETTh2.csv",
        "data_name": "ETTh2",
        "seq_len": 336,
        "enc_in": 7,
        "batch_size": 64,
        "pred_lens": [96, 192, 336, 720],
        "aug_types": ["Adaptive-Channel-Preserve"],
        "aug_params": {
            96: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            192: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            336: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 36,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
                },
            },
            720: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 48,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
                },
            },
        },
    },
    "weather": {
        "data_path": "./dataset/weather.csv",
        "data_name": "weather",
        "seq_len": 336,
        "enc_in": 21,
        "batch_size": 64,
        "pred_lens": [96, 192, 336, 720],
        "aug_types": ["Adaptive-Channel-Preserve"],
        "aug_params": {
            96: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            192: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 24,
                    "mix_rate": 0.15,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 1.0,
                },
            },
            336: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 36,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 1.0,
                },
            },
            720: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 48,
                    "mix_rate": 0.1,
                    "scale_factor": 0.05,
                    "variance_threshold": 0.03,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.7,
                },
            },
        },
    },
    "ILI": {
        "data_path": "./dataset/national_illness.csv",
        "data_name": "custom",
        "seq_len": 36,
        "enc_in": 7,
        "batch_size": 32,
        "pred_lens": [24, 36, 48, 60],
        "aug_types": ["Adaptive-Channel-Preserve"],
        "aug_params": {
            24: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 6,
                    "mix_rate": 0.1,
                    "scale_factor": 0.03,
                    "variance_threshold": 0.02,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            36: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 6,
                    "mix_rate": 0.1,
                    "scale_factor": 0.03,
                    "variance_threshold": 0.02,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            48: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 6,
                    "mix_rate": 0.1,
                    "scale_factor": 0.03,
                    "variance_threshold": 0.02,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
            60: {
                "Adaptive-Channel-Preserve": {
                    "segment_size": 6,
                    "mix_rate": 0.1,
                    "scale_factor": 0.03,
                    "variance_threshold": 0.02,
                    "corr_threshold": 0.7,
                    "sampling_rate": 0.5,
                },
            },
        },
    },
}
