import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def nrmse(pred, target):
    """Normalized RMSE"""
    return torch.sqrt(torch.mean((pred - target) ** 2)) / torch.std(target)


def r2_score(pred, target):
    """R-squared metric"""
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ss_res / ss_tot


def accuracy_metric(pred, target, tol=0.1):
    """percentage predictions within tolerance"""
    error = torch.abs(pred - target)
    return torch.mean((error < tol).float())


def prepare_data(cfg):
    

    df= pd.read_excel("ESN8.xlsx")
    height = torch.tensor(df["Lan Height"].values, dtype=torch.float32)
    flow = torch.tensor(df["V of Oxy"].values, dtype=torch.float32)
    target = torch.tensor(df["dc/dt"].values, dtype=torch.float32)
    time = torch.tensor(df["Time"].values, dtype=torch.float32)
    


    inputs = torch.stack([height, flow,time], dim=1)
    targets = target.unsqueeze(-1)
    

    train_len = cfg['train_len']
    test_len = cfg['test_len']

    train_in = inputs[:train_len]
    train_out = targets[1:train_len+1]

    test_in = inputs[train_len:train_len+test_len]
    test_out = targets[train_len+1:train_len+test_len+1]

    mean = train_in.mean(0)
    std = train_in.std(0)

    train_in = (train_in - mean) / std
    test_in = (test_in - mean) / std

    t_mean = train_out.mean()
    t_std = train_out.std()

    train_out = (train_out - t_mean) / t_std
    test_out = (test_out - t_mean) / t_std

    return train_in, train_out, test_in, test_out, t_mean, t_std,mean, std



def load_full_data():
    """Load entire dataset for full-curve prediction (visualization only)"""

    df = pd.read_excel("ESN8.xlsx")

    height = torch.tensor(df["Lan Height"].values, dtype=torch.float32)
    flow = torch.tensor(df["V of Oxy"].values, dtype=torch.float32)
    target = torch.tensor(df["dc/dt"].values, dtype=torch.float32)
    time = torch.tensor(df["Time"].values, dtype=torch.float32)

    inputs = torch.stack([height, flow,time], dim=1)
    targets = target.unsqueeze(-1)

    

    return inputs, targets



def print_cfg(cfg):
    print(
        f"ESN:\n"
        f"res={cfg['n_res']}, rho={cfg['rho']}, dens={cfg['density']}\n"
        f"wash={cfg['washout']}, train={cfg['train_len']}, test={cfg['test_len']}\n"
    )


def save_predictions_csv(pred, target, filename="results.csv"):

    pred = pred.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()

    df = pd.DataFrame({
        "True_dc_dt": target,
        "Predicted_dc_dt": pred
    })

    df.to_csv(filename, index=False)

    print(f"Saved predictions to {filename}")



def plot_predictions(pred, target,filename="prediction_plot.png"):

    pred = pred.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()

    plt.figure(figsize=(10,5))

    plt.plot(target, label="True", linewidth=2)
    plt.plot(pred, label="Predicted", linestyle="--")

    plt.xlabel("Time step")
    plt.ylabel("dc/dt")
    plt.title("ESN Prediction vs True Signal")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    

    print(f"Saved plot to {filename}")  
    plt.close()
  