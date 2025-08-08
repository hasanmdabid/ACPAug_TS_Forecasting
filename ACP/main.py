import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.dataloader import TimeSeriesDataset
from src.model import DLinear
from src.dataset_parameter import (
    dataset_configs,
    epochs,
    learning_rate,
    patience,
    num_iterations,
    label_len,
)
from src.train_eval import train, validate, test

# Main experiment
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    # Run experiments for each dataset
    for dataset_name, config in dataset_configs.items():
        print(f"\n=== Processing dataset: {dataset_name} ===")
        results = {}
        # Load datasets
        for pred_len in config["pred_lens"]:
            print(f"\nPrediction length: {pred_len}")
            train_dataset = TimeSeriesDataset(
                data_path=config["data_path"],
                data_name=config["data_name"],
                flag="train",
                seq_len=config["seq_len"],
                label_len=label_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
            )
            val_dataset = TimeSeriesDataset(
                data_path=config["data_path"],
                data_name=config["data_name"],
                flag="val",
                seq_len=config["seq_len"],
                label_len=label_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
            )
            test_dataset = TimeSeriesDataset(
                data_path=config["data_path"],
                data_name=config["data_name"],
                flag="test",
                seq_len=config["seq_len"],
                label_len=label_len,
                pred_len=pred_len,
                enc_in=config["enc_in"],
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=True,
            )

            # Run experiment for Adaptive-Channel-Preserve
            aug_type = "Adaptive-Channel-Preserve"
            params = config["aug_params"][pred_len][aug_type]
            mse_list, mae_list, rse_list, val_loss_list = [], [], [], []
            print(
                f"\nRunning experiment with {aug_type} augmentation for {dataset_name}, pred_len={pred_len}..."
            )
            for itr in range(num_iterations):
                print(f"Iteration {itr+1}/{num_iterations}")
                model = DLinear(
                    config["seq_len"],
                    pred_len,
                    enc_in=config["enc_in"],
                    individual=False,
                ).to(device)
                val_loss = train(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    config["seq_len"],
                    label_len,
                    pred_len,
                    segment_size=params["segment_size"],
                    mix_rate=params["mix_rate"],
                    scale_factor=params["scale_factor"],
                    variance_threshold=params["variance_threshold"],
                    corr_threshold=params["corr_threshold"],
                    sampling_rate=params["sampling_rate"],
                    epochs=epochs,
                    lr=learning_rate,
                    patience=patience,
                )
                model.load_state_dict(
                    torch.load(f"./checkpoints/{aug_type}/checkpoint.pth")
                )
                mae, mse, rse = test(
                    model, test_loader, device, train_dataset.scaler, pred_len
                )
                mse_list.append(mse)
                mae_list.append(mae)
                rse_list.append(rse)
                val_loss_list.append(val_loss)
                print(
                    f"Iteration {itr+1} - Val Loss: {val_loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}, RSE: {rse:.6f}"
                )

            results[(pred_len, aug_type)] = {
                "val_loss": np.mean(val_loss_list),
                "mae": np.mean(mae_list),
                "mse": np.mean(mse_list),
                "rse": np.mean(rse_list),
                "mae_std": np.std(mae_list),
                "mse_std": np.std(mse_list),
                "rse_std": np.std(rse_list),
            }
            print(
                f'{aug_type} - Avg Val Loss: {results[(pred_len, aug_type)]["val_loss"]:.6f}, Avg MAE: {results[(pred_len, aug_type)]["mae"]:.6f}, '
                f'Avg MSE: {results[(pred_len, aug_type)]["mse"]:.6f}, Avg RSE: {results[(pred_len, aug_type)]["rse"]:.6f}, '
                f'MSE Std: {results[(pred_len, aug_type)]["mse_std"]:.6f}'
            )

            # Plot predictions
            model.load_state_dict(
                torch.load(f"./checkpoints/{aug_type}/checkpoint.pth")
            )
            model.eval()
            with torch.no_grad():
                batch_x, batch_y = next(iter(test_loader))
                batch_x = batch_x.float().to(device)
                outputs = model(batch_x)
                outputs = outputs[:, -pred_len:, :].cpu().numpy()
                batch_y = batch_y[:, -pred_len:, :].cpu().numpy()
                outputs = train_dataset.scaler.inverse_transform(outputs[0])
                batch_y = train_dataset.scaler.inverse_transform(batch_y[0])
                plt.figure()
                plt.plot(batch_y[:, 0], label="Ground Truth")
                plt.plot(outputs[:, 0], label=f"Prediction ({aug_type})")
                plt.legend()
                plt.savefig(
                    f"./plots/prediction_{dataset_name}_{aug_type}_{pred_len}.png"
                )
                plt.close()

        # Save results
        with open(f"results_{dataset_name}_adaptive_channel_preserve.csv", "w") as f:
            for pred_len in config["pred_lens"]:
                for aug_type in config["aug_types"]:
                    metrics = results[(pred_len, aug_type)]
                    f.write(
                        f"{dataset_name}, pred_len={pred_len}, {aug_type}: "
                        f'Val Loss={metrics["val_loss"]:.6f}, MAE={metrics["mae"]:.6f}, MSE={metrics["mse"]:.6f}, '
                        f'RSE={metrics["rse"]:.6f}, MAE Std={metrics["mae_std"]:.6f}, MSE Std={metrics["mse_std"]:.6f}, '
                        f'RSE Std={metrics["rse_std"]:.6f}\n'
                    )
