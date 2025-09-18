import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier


def get_models(model_configs: dict, rnd_state: int) -> dict:
    models = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "SVM": SVC,
        "LightGBM": LGBMClassifier,
    }

    initialized_models = {}
    for name, params in model_configs.items():
        if name in models:
            # Add common parameters like random_state or probability=True for SVC
            if "random_state" in models[name]().get_params():
                params['random_state'] = rnd_state
            if name == "SVM" and 'probability' not in params:
                params['probability'] = True  # Needed for roc_auc_score with probabilities

            initialized_models[name] = models[name](**params)

    return initialized_models