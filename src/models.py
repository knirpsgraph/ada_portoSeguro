import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

def get_models(C_logreg, RND):
    return {
        "LogisticRegression": LogisticRegression(penalty="l2", solver="saga", C=C_logreg, class_weight="balanced", max_iter=2000, random_state=RND),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=RND),
        "SVM": SVC(kernel='rbf', random_state=RND, probability=True), # probability=True is needed for roc_auc_score
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0, random_state=RND, n_jobs=-1)
    }