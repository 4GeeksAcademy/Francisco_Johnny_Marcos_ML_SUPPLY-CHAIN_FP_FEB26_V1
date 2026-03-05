# src/utils.py

# ==========================================
# 1. SYSTEM & FILE MANAGEMENT
# ==========================================
import os
import json
import math
import joblib
import pickle
import requests
from pathlib import Path

# ==========================================
# 2. DATA MANIPULATION & STORAGE
# ==========================================
import pandas as pd
import numpy as np
from numpy._core.defchararray import upper
from sqlalchemy import create_engine
import pyarrow
import fastparquet
from tabulate import tabulate

# ==========================================
# 3. GEOSPATIAL & VISUALIZATION
# ==========================================
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 4. MACHINE LEARNING - PREPROCESSING
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest, f_regression
from sklearn.feature_extraction.text import CountVectorizer

# ==========================================
# 5. MACHINE LEARNING - MODELS (DATACO PROJECT)
# ==========================================
# Supervised: Classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from xgboost import XGBClassifier

# Supervised: Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV

# ==========================================
# 6. MODEL EVALUATION
# ==========================================
from sklearn.metrics import (
    accuracy_score, 
    r2_score, 
    mean_absolute_error, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)