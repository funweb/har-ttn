parameters_dict = {
    "learning_rate_1": 0.0001,
    "num_train": 8000,
    # "numBatches": num_train / batch_size,
    "dataset_names": ["cairo", "milan", "kyoto7", "kyoto8", "kyoto11"],
    "database_dir": "../datasets/casas",
    "ksplit": 3,

    "num_classes": "",  #TODO: 类别数量, 根据需要设置
    "batch_size": 32,
    "maxIters": 10000,
    "seq_len": 100,
    "distance_int": "",
    "dataset_name": "",
}