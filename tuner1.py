import copy
import itertools
import torch

from esn import ESNLayer
from utils import prepare_data, nrmse, r2_score


def tune_esn(cfg):

    print("\n🔍 Starting hyperparameter tuning...\n")

    # -------- search space --------
    grid = {
        "n_res": [5,10, 15, 20, 30],
        "leak_rate": [0.1,0.3, 0.5, 0.7],
        "rho": [0.5,0.7, 0.9, 1.0],
        "reg": [1e-3, 1e-2, 1e-1],
        "density": [0.1, 0.2]
    }

    keys = grid.keys()
    combinations = list(itertools.product(*grid.values()))

    best_score = -1e9
    best_cfg = None

    train_in, train_out, test_in, test_out,*rest = prepare_data(cfg)

    for values in combinations:

        temp_cfg = copy.deepcopy(cfg)

        for k, v in zip(keys, values):
            temp_cfg[k] = v

        esn = ESNLayer(
            n_in=3,
            n_res=temp_cfg['n_res'],
            n_out=1,
            spectral_radius=temp_cfg['rho'],
            density=temp_cfg['density'],
            input_scale=temp_cfg['input_scale'],
            leak_rate=temp_cfg['leak_rate'],
            reg=temp_cfg['reg'],
            seed=temp_cfg['seed'],
            device=temp_cfg['device']
        )

        esn.fit(train_in, train_out, washout=temp_cfg['washout'])

        pred = esn.predict(test_in)

        score = r2_score(pred, test_out).item()

        print(f"Tested: {temp_cfg}  →  R2 = {score:.4f}")

        if score > best_score:
            best_score = score
            best_cfg = temp_cfg

    print("\n✅ BEST CONFIG FOUND:")
    print(best_cfg)
    print(f"Best R2 = {best_score:.4f}\n")

    return best_cfg
