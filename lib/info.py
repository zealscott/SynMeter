# this file contains the basic configuration of the project

# root directory
ROOT_DIR = "/home/ytdu/SynMeter"

# tuned parameters path
TUNED_PARAMS_PATH = ROOT_DIR + "/exp"

# number of trials for tuning
NUMS_TRIALS = 50

# number of samples for evaluation
N_EXPS = 10

# minimum number per class
MIN_NUM_PER_CLASS = 10

# sqlite database path for optuna tuning
STORAGE = "sqlite:///exp.db"

# ml evaluators
EVALUATOR = [
    "svm",
    "lr",
    "tree",
    "rf",
    "xgboost",
    "cat_boost",
    "mlp",
    "tab_transformer",
]


# datasets
DATASETS = [
    "adult",
    "shoppers",
    "phishing",
    "magic",
    "faults",
    "bean",
    "obesity",
    "robot",
    "abalone",
    "news",  # cannot be processed by great
    "insurance",
    "wine",
]

ALGORITHMS = ["pgm", "privsyn", "tvae", "ctgan", "tabddpm", "great", "pategan_eps_1", "tablediffusion_eps_1"]
