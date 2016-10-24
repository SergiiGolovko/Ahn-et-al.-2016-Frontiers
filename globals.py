# define different run configurations and some global parameters
# config normal
CONFIG_normal = {'CONFIG': 'config_normal',
                 'N_FOLDS': 5,
                 'N_REPLICATIONS': 100,
                 'N_FOLD_SPLITS': 1000,
                 'SURVIVAL_RATE_CUTOFF': 0.05,
                 'TEST_SIZE': 0.33,
                 'ESTIMATION_MODE': 'single'  # if single -> only one train/test split is created
                                                 # if replicate -> N_REPLICATIONS train/test splits are created
                                                 # if both -> first run single, followed by replicate
                 }

# config test
CONFIG_test = CONFIG_normal.copy()
CONFIG_test['CONFIG'] = 'config_test'
CONFIG_test['N_REPLICATIONS'] = 10
CONFIG_test['N_FOLD_SPLITS'] = 100

# select a configuration to run
CONFIG = CONFIG_normal

# in order to run 'normal', just comment out the following line, do NOT delete it
CONFIG = CONFIG_test

# random seeds
RANDOM_SEED = 2016
RANDOM_SEED_MIN = 0
RANDOM_SEED_MAX = 56000

# hyperopt max evaluations
HYPEROPT_MAX_EVALS = 100
