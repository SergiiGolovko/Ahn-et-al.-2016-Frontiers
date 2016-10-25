# define different run configurations and some global parameters

# Default parameters and running time information. Running time may be different for different computers. Check ETA time
# in progressbar!
# Important - to reproduce figures 1 and 2: set ESTIMATION_MODE = 'single'
#                                           when N_FOLD_SPLITS = 1000 runs about 40 min
#             to reproduce figure 4: set ESTIMATION_MODE = 'replicate'
#                                           when N_REPLICATIONS = 100, N_FOLD_SPLITS = 10 runs about 20 min
#                                           when N_REPLICATIONS = 1000, N_FOLD_SPLITS = 10 runs about 3 hours

# config normal
CONFIG_normal = {'CONFIG': 'config_normal',
                 'N_FOLDS': 5,                   # number of folds for each fold split and N_REPLICATIONS to be run
                 'N_REPLICATIONS': 100,          # number of replications, ignored if ESTIMATION_MODE = 'single'
                 'N_FOLD_SPLITS': 1000,           # number of fold splits
                 'SURVIVAL_RATE_CUTOFF': 0.05,   # survival cutoff rate for beta coefficients,
                 'TEST_SIZE': 0.33,              # the percentage of data to be used as a test set, ignored if
                                                 # ESTIMATION_MODE = 'single'
                 'ESTIMATION_MODE': 'single'     # if single -> only one train/test split is created
                                                 # if replicate -> N_REPLICATIONS train/test splits are created
                                                 # if both -> first run single, followed by replicate
                 }

# config test
CONFIG_test = CONFIG_normal.copy()
CONFIG_test['CONFIG'] = 'config_test'
CONFIG_test['ESTIMATION_MODE'] = 'single'
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
