dataset_configs = {
    "ETTh1": {
        "data_path": "/home/abidhasan/Documnet/Project/a_c_p/dataset/ETTh1.csv",
        "data_name": "ETTh1",
        "seq_len": 336,
        "pred_lens": [96, 192, 336, 720],
        "enc_in": 7,
        "batch_size": 32,
        "aug_types": ["Wave-Freq"],
        "aug_params": {
            96: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            192: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            336: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            720: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
        },
    },
    "ETTh2": {
        "data_path": "/home/abidhasan/Documnet/Project/a_c_p/dataset/ETTh2.csv",
        "data_name": "ETTh2",
        "seq_len": 336,
        "pred_lens": [96, 192, 336, 720],
        "enc_in": 7,
        "batch_size": 32,
        "aug_types": ["Wave-Freq"],
        "aug_params": {
            96: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            192: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            336: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
            720: {
                "Wave-Freq": {
                    "mask_rate": 0.3,
                    "wavelet": "db2",
                    "level": 3,
                    "sampling_rate": 0.2,
                }
            },
        },
    },
    # Add other datasets as needed
}
