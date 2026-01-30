from esn import ESNLayer
from utils import nrmse, r2_score, accuracy_metric, prepare_data, print_cfg
from utils import save_predictions_csv
from utils import plot_predictions
from utils import load_full_data
from tuner import tune_esn
import matplotlib
matplotlib.use('Agg')  # non-GUI backend






def config():
    """General parameter sweep tests on Mackey-glass yielded the following configurations:"""
    return {                    # | Suggested   | Best Found   | Top 10% Range     |
        'n_res': 15,           
        'rho': 0.92,            
        'density': 0.17,        
        'input_scale': 3.2,     
        'leak_rate': 0.79,      
        'reg': 0.01,           
        'washout': 5,         
        'train_len': 120,     
        'test_len': 20,       
        'seed': 1001,
        'device': 'cpu'
    }


def train_esn(cfg, train_in, train_out):
    esn = ESNLayer(n_in=2, n_res=cfg['n_res'], n_out=1, spectral_radius=cfg['rho'], density=cfg['density'], input_scale=cfg['input_scale'], leak_rate=cfg['leak_rate'], reg=cfg['reg'], seed=cfg['seed'], device=cfg['device'])
    esn.fit(train_in, train_out, washout=cfg['washout'])
    return esn


def evaluate_esn(esn, test_in, test_out, t_mean, t_std):

    pred_scaled = esn.predict(test_in)

    
    pred = pred_scaled * t_std + t_mean
    target = test_out * t_std + t_mean

    error = nrmse(pred, target)
    r2 = r2_score(pred, target)
    acc = accuracy_metric(pred, target)

    save_predictions_csv(pred, target, "prediction_results.csv")

    return pred, error, r2, acc





def main():

    cfg = config()
    

    cfg = tune_esn(cfg)
    print_cfg(cfg)


    train_in, train_out, test_in, test_out, t_mean, t_std, in_mean, in_std = prepare_data(cfg)

    
    



    esn = train_esn(cfg, train_in, train_out)

    pred, error, r2, acc = evaluate_esn(esn, test_in, test_out, t_mean, t_std)


    # -------- Test set plot (metrics) --------
    


    


    print(f"NRMSE: {error:.6f}")
    print(f"R2 score: {r2:.4f}")
    print(f"Accuracy: {acc*100:.2f}%")


    print('1')
    full_in, full_out = load_full_data()

    # scale inputs using TRAIN stats
    full_in = (full_in - in_mean) / in_std

    print('2')

    # 1️⃣ predict FIRST
    pred_full_scaled = esn.predict(full_in)

    # 2️⃣ inverse transform
    pred_full = pred_full_scaled * t_std + t_mean
    full_out_real = full_out

    print('3')

    # 3️⃣ save
    save_predictions_csv(pred_full, full_out_real, "full_prediction_results.csv")

    print('4')

    # 4️⃣ plot
    plot_predictions(pred_full, full_out_real, "full_prediction_plot.png")

    print('5')

    test_out_real = test_out * t_std + t_mean
    plot_predictions(pred, test_out_real,"test_prediction_plot.png")


    


if __name__ == "__main__":
    main()