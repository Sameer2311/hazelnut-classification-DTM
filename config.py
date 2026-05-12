# config.py - MASTER CONFIGURATION FILE
# Everyone commits their changes here

# ============================================
# DEFAULT CONFIGURATION (DO NOT CHANGE)
# ============================================
DEFAULT_CONFIG = {
    'epochs': 10,
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'adam',
    'dropout_rate': 0.0,
    'hidden_layers': [128, 64],
}

# ============================================
# PERSON 1: LEARNING RATE EXPERIMENTS
# ============================================
LR_CONFIGS = {
    'baseline': 0.001,
    'experiment_1': 0.01,
    'experiment_2': 0.0001,
    'experiment_3': 0.0005,
}

# ============================================
# PERSON 2: BATCH SIZE EXPERIMENTS  
# ============================================
BATCH_CONFIGS = {
    'baseline': 32,
    'experiment_1': 16,
    'experiment_2': 64,
    'experiment_3': 128,
}

# ============================================
# PERSON 3: LAYER CONFIGURATIONS
# ============================================
LAYER_CONFIGS = {
    'baseline': [128, 64],
    'experiment_1': [64, 32],      # Smaller network
    'experiment_2': [256, 128],    # Larger network
    'experiment_3': [256, 128, 64], # Deeper network
}

# ============================================
# PERSON 4: OPTIMIZER CONFIGURATIONS
# ============================================
OPTIMIZER_CONFIGS = {
    'baseline': 'adam',
    'experiment_1': 'sgd',
    'experiment_2': 'rmsprop',
    'experiment_3': 'adamw',
}

# ============================================
# PERSON 5: DROPOUT RATES
# ============================================
DROPOUT_CONFIGS = {
    'baseline': 0.0,
    'experiment_1': 0.2,
    'experiment_2': 0.4,
    'experiment_3': 0.6,
}