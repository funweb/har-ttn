parameters_dict = {
    "learning_rate_1": 0.0001,
    # "numBatches": num_train / batch_size,
    "dataset_names": ["cairo", "milan", "kyoto7", "kyoto8", "kyoto11"],
    "database_dir": "../datasets/casas",
    "ksplit": 3,

    "method_name": "ttn",

    "num_classes": "",  #TODO: 类别数量, 根据需要设置
    "num_train": "",
    "batch_size": 32,
    "maxIters": 10000,
    "seq_len": "",
    "distance_int": "",
    "dataset_name": "",
}

# jupyter
JUPYTER_TOKEN = {
    "token_lists": ["root",  # win
                    "8786ba7fd6db486eb13a6e4e79d5951a",  # 7920
                    "ef6c4f2832454a65a1d9c2c7551af431"]  # 5810
}