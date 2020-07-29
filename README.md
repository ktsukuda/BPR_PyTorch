# BPR_PyTorch

Bayesian Personalized Ranking with PyTorch.

## Environment

- Python: 3.6
- PyTorch: 1.5.1
- CUDA: 10.1
- Ubuntu: 18.04

## Dataset

[The Movielens 1M Dataset](http://grouplens.org/datasets/movielens/1m/) is used. The rating data is included in [data/ml-1m](https://github.com/ktsukuda/BPR_PyTorch/tree/master/data/ml-1m).

## Run the Codes

```bash
$ python BPR_PyTorch/main.py
```

## Details

For each user, the latest and the second latest rating are used as test and validation, respectively. The remaining ratings are used as training. The hyperparameters (batch_size, lr, latent_dim, l2_reg) are tuned by using the valudation data in terms of nDCG. See [config.ini](https://github.com/ktsukuda/BPR_PyTorch/blob/master/BPR_PyTorch/config.ini) about the range of each hyperparameter.

Although the original ratings range 1 to 5, all of them are used as positive data. Items that are not consumed by a user are used as negative data for the user.

By running the code, hyperparameters are automatically tuned. After the training process, the best hyperparameters and HR/nDCG computed by using the test data are displayed.

Given a specific combination of hyperparameters, the corresponding training results are saved in `data/train_result/<hyperparameter combination>` (e.g., data/train_result/batch_size_512-lr_0.005-latent_dim_8-l2_reg_1e-07-epoch_3-n_negative_4-top_k_10). In the directory, a model file (`model.pth`) and a json file (`epoch_data.json`) that describes information for each epoch are generated. The json file can be described as follows (epoch=3).

```json
[
    {
        "epoch": 0,
        "loss": 4994338.738952637,
        "HR": 0.0728476821192053,
        "NDCG": 0.03107992452383637
    },
    {
        "epoch": 1,
        "loss": 3138874.3325195312,
        "HR": 0.14966887417218544,
        "NDCG": 0.06523117735080695
    },
    {
        "epoch": 2,
        "loss": 1708930.3295593262,
        "HR": 0.4064569536423841,
        "NDCG": 0.21713509086237084
    }
]
```
