import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import DLinear, iTransformer
from train_eval import train, test
from dataloader import TimeSeriesDataset
from dataset_parameter import dataset_configs
import gc
import pandas as pd

# Main experiment
def main(epochs, learning_rate, patience, num_iterations, label_len):
    # Common parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)  # Limit to 50% of GPU memory
    models = ["iTransformer"]

    # Create directories
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Run experiments for each dataset
    for dataset_name, config in dataset_configs.items():
        # Initialize results file with header
        results_file = f"/home/abidhasan/Documnet/Project/a_c_p/baseline_papers/results/results_{dataset_name}.csv"
        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                f.write(
                    "Dataset,Model,Pred_Len,Aug_Type,Val_Loss,MAE,MSE,RSE,MAE_Std,MSE_Std,RSE_Std\n"
                )

        results = {}
        for model_name in models:
            print(
                f"\n=== Processing dataset: {dataset_name} with model: {model_name} ==="
            )
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
                    n_imf=max(
                        [
                            config["aug_params"][pred_len][aug_type]["n_imf"]
                            for aug_type in config["aug_types"]
                        ]
                    ),
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

                # Run experiments for each augmentation type
                for aug_type in config["aug_types"]:
                    params = config["aug_params"][pred_len][aug_type]
                    mse_list, mae_list, rse_list = [], [], []
                    print(
                        f"\nRunning experiment with {model_name} model, {aug_type} augmentation for {dataset_name}, pred_len={pred_len}..."
                    )
                    for itr in range(num_iterations):
                        print(f"Iteration {itr+1}/{num_iterations}")
                        if model_name == "DLinear":
                            model = DLinear(
                                config["seq_len"],
                                pred_len,
                                enc_in=config["enc_in"],
                                individual=False,
                            ).to(device)
                        elif model_name == "iTransformer":
                            model = iTransformer(
                                seq_len=config["seq_len"],
                                pred_len=pred_len,
                                enc_in=config["enc_in"],
                                d_model=512,
                                n_heads=8,
                                e_layers=4,
                                d_ff=2048,
                                dropout=0.1,
                            ).to(device)
                        val_loss = train(
                            model,
                            train_loader,
                            val_loader,
                            device,
                            aug_type,
                            config["seq_len"],
                            label_len,
                            pred_len,
                            aug_rate=params["aug_rate"],
                            rates=params["rates"],
                            wavelet=params["wavelet"],
                            level=params["level"],
                            sampling_rate=params["sampling_rate"],
                            n_imf=params["n_imf"],
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
                        print(
                            f"Iteration {itr+1} - Val Loss: {val_loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}, RSE: {rse:.6f}"
                        )

                    results[(pred_len, aug_type)] = {
                        "val_loss": np.mean(
                            [
                                train(
                                    model,
                                    train_loader,
                                    val_loader,
                                    device,
                                    aug_type,
                                    config["seq_len"],
                                    label_len,
                                    pred_len,
                                    aug_rate=params["aug_rate"],
                                    rates=params["rates"],
                                    wavelet=params["wavelet"],
                                    level=params["level"],
                                    sampling_rate=params["sampling_rate"],
                                    n_imf=params["n_imf"],
                                    epochs=epochs,
                                    lr=learning_rate,
                                    patience=patience,
                                )
                                for _ in range(num_iterations)
                            ]
                        ),
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

                    # Save results after each experiment
                    metrics = results[(pred_len, aug_type)]
                    with open(results_file, "a") as f:
                        f.write(
                            f"{dataset_name},{model_name},{pred_len},{aug_type},"
                            f'{metrics["val_loss"]:.6f},{metrics["mae"]:.6f},{metrics["mse"]:.6f},'
                            f'{metrics["rse"]:.6f},{metrics["mae_std"]:.6f},{metrics["mse_std"]:.6f},'
                            f'{metrics["rse_std"]:.6f}\n'
                        )

                    # Plot predictions
                    model.load_state_dict(
                        torch.load(f"./checkpoints/{aug_type}/checkpoint.pth")
                    )
                    model.eval()
                    with torch.no_grad():
                        batch_x, batch_y, _ = next(iter(test_loader))
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
                    # Clean up memory after aug_type
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

                # Clean up datasets after pred_len
                del (
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    train_loader,
                    val_loader,
                    test_loader,
                )
                torch.cuda.empty_cache()
                gc.collect()

    print("All experiments completed.")

if __name__ == "__main__":
    main(epochs=30, learning_rate=0.01, patience=10, num_iterations=5, label_len=0)
    print("Main experiment finished.")
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleanup completed.")
    print("You can now check the results and plots in the respective directories.")
