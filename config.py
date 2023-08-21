config = {
    "epochs": 20,
    "batch_size": 32,
    "input_nodes": 28 * 28 * 1,
    "output_nodes": 512,
    "num_classes": 10,
    "input_shape": (28, 28, 1),
    "flatten_shape": 28 * 28 * 1,
    "min_node": 0,
    "max_node": 1,

    "learning_rate": 0.01,
    "num_training_data": 50000,

    "dataset_name": "mnist",
    # "dataset_name": "fashion_mnist",
    # "dataset_name": "kmnist",

    "use_bias": True,
    # "use_bias": False,

    ##############################
    ##          PATTERN         ##
    ##############################
    # --- Either to use pattern layer or not ---
    "use_pattern": True,
    # "use_pattern": False,

    #--- Either to use a fixed pattern for each class or use different pattern for different classes ---
    "num_of_pattern_per_label": 1,
    "use_same_pattern": True,
    "use_same_pattern": False,

    # --- Either to use a trainable (dynamic) pattern for each class or not ---
    "use_flexible_pattern": True,
    # "use_flexible_pattern": False,


    # --- Type of loss function ---
    "loss": "mse",
    # "loss": "kl",

    # --- Type of loss function ---
    "activation": False,
    # "activation": "sigmoid",
    # "activation": "tanh",
}