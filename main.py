import argparse
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from datasets import ECL, Climate, ETTh1, ETTh2, Power
from models import DecisionTreeME
from metrics import Metric, seed_torch, reduce_mem_usage

def parse_args():
    parser = argparse.ArgumentParser(description='Decision Tree with Lasso Experts')
    parser.add_argument('--dataset', type=str, default='ECL', help='Dataset name')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction length')
    parser.add_argument('--min_samples', type=int, default=200, help='Minimum samples per node')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum tree depth')
    parser.add_argument('--alpha', type=float, default=0.1, help='Lasso alpha')
    parser.add_argument('--ut_hp', type=float, default=0.1, help='Expert utilization hyperparameter')
    parser.add_argument('--n_thresholds', type=int, default=1000, help='Number of thresholds to evaluate per feature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def load_dataset(args):
    dataset_map = {
        'ECL': ECL,
        'Climate': Climate,
        'ETTh1': ETTh1,
        'ETTh2': ETTh2,
        'Power': Power
    }
    if args.dataset not in dataset_map:
        raise NotImplementedError(f"No such dataset: {args.dataset}")
    
    # Ajusta los paths a tu repositorio
    root_paths = {
        'ECL': 'data/ECL',
        'Climate': 'data/Climate',
        'ETTh1': 'data/ETTh1',
        'ETTh2': 'data/ETTh2',
        'Power': 'data/Power'
    }
    
    dataset_cls = dataset_map[args.dataset]
    dataset = dataset_cls(root_path=root_paths[args.dataset],
                          seq_len=24,
                          pred_len=args.pred_len,
                          features='S',
                          scale=True,
                          num_ts=1)
    return dataset

def main():
    args = parse_args()
    np.random.seed(args.seed)
    seed_torch(args.seed)

    dataset = load_dataset(args)

    X_train, y_train = dataset.train_x.reshape(dataset.train_x.shape[0], dataset.train_x.shape[1]), dataset.train_y.reshape(dataset.train_y.shape[0], dataset.train_y.shape[1])
    X_val, y_val = dataset.valid_x.reshape(dataset.valid_x.shape[0], dataset.valid_x.shape[1]), dataset.valid_y.reshape(dataset.valid_y.shape[0], dataset.valid_y.shape[1])
    X_test, y_test = dataset.test_x.reshape(dataset.test_x.shape[0], dataset.test_x.shape[1]), dataset.test_y.reshape(dataset.test_y.shape[0], dataset.test_y.shape[1])

    tree = DecisionTreeME(min_samples=args.min_samples, max_depth=args.max_depth)

    tree.fit(X_train, y_train, X_val, y_val, alpha=args.alpha, ut_hp=args.ut_hp)

    full_dataset = np.concatenate((np.concatenate((X_train, y_train), axis=1),
                                   np.concatenate((X_val, y_val), axis=1)), axis=0)
    tree.retrain_leaf_nodes(full_dataset, node=tree.root, param_grid={'alpha': np.logspace(np.log10(args.alpha), 0, 10)}, cv=5)


    pred_test = tree.predict(X_test)


    mae, mse, rmse, mape, mspe = Metric(np.array(pred_test).reshape(-1, 1), y_test)
    result = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'MAPE': [mape],
        'MSPE': [mspe]
    })
    print(result)

if __name__ == "__main__":
    main()
