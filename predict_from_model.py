"""
UPDATED: Generate predictions WITH auto-calculated injuries, form, H2H
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
CSV_FILES = ["teams_2024.csv", "teams_2025.csv"]
TARGET_COLUMN = "xGoalsPercentage_5on5"
MODEL_PATH = "models/m3_local_model.pkl"

BASE_STATS = [
    "xGoalsPercentage", "corsiPercentage", "fenwickPercentage",
    "goalsFor", "goalsAgainst", "shotsOnGoalFor", "shotsOnGoalAgainst",
    "highDangerGoalsFor", "highDangerGoalsAgainst", "reboundsFor", "reboundsAgainst"
]

SITUATIONS = ["5on5", "5on4", "4on5", "4on4"]

def load_data():
    frames = []
    for filename in CSV_FILES:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(df)
    
    if not frames:
        raise FileNotFoundError(f"No CSV files found")
    
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined['situation'].isin(SITUATIONS)].copy()
    
    pivoted = combined.pivot_table(
        index=['team', 'season'],
        columns='situation',
        values=BASE_STATS,
        aggfunc='first'
    )
    
    pivoted.columns = ['_'.join(col) for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    return pivoted

def engine_features(df, injuries_dict, form_dict):
    df = df.copy()
    
    df["goal_diff_5on5"] = df["goalsFor_5on5"] - df["goalsAgainst_5on5"]
    df["shot_diff_5on5"] = df["shotsOnGoalFor_5on5"] - df["shotsOnGoalAgainst_5on5"]
    df["high_danger_diff_5on5"] = df["highDangerGoalsFor_5on5"] - df["highDangerGoalsAgainst_5on5"]
    df["shooting_pct_5on5"] = df["goalsFor_5on5"] / (df["shotsOnGoalFor_5on5"] + 1)
    df["save_pct_5on5"] = 1 - (df["goalsAgainst_5on5"] / (df["shotsOnGoalAgainst_5on5"] + 1))
    
    df['injury_penalty'] = df['team'].map(injuries_dict).fillna(0)
    df['recent_form'] = df['team'].map(form_dict).fillna(0)
    df['h2h_vs_opponent'] = 0
    
    return df

def load_helper_data():
    injuries_dict = {}
    form_dict = {}
    
    injuries_path = os.path.join(DATA_DIR, "injuries.csv")
    if os.path.exists(injuries_path):
        injuries = pd.read_csv(injuries_path)
        injuries_dict = injuries.groupby('team')['injury_impact'].sum().to_dict()
        logger.info(f"✓ Loaded auto-calculated injuries for {len(injuries_dict)} teams")
    
    form_path = os.path.join(DATA_DIR, "form.csv")
    if os.path.exists(form_path):
        form = pd.read_csv(form_path)
        form_dict = (form.set_index('team')['last_10_win_pct'] - 0.5).to_dict()
        logger.info(f"✓ Loaded form for {len(form_dict)} teams")
    
    return injuries_dict, form_dict

def prepare_data_for_prediction(df, feature_names, injuries_dict, form_dict):
    df = engine_features(df, injuries_dict, form_dict)
    
    for col in feature_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    X = df[feature_names].values
    
    return X, df

def main():
    logger.info("="*60)
    logger.info("PREDICTION PIPELINE (With Auto-Calculated Features)")
    logger.info("="*60)
    
    logger.info("\nLoading model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    logger.info(f"✓ Loaded model with {len(feature_names)} features")
    
    logger.info("\nLoading data...")
    df = load_data()
    logger.info(f"✓ Loaded {len(df)} team-seasons")
    
    logger.info("\nLoading auto-calculated helper data...")
    injuries_dict, form_dict = load_helper_data()
    
    logger.info("\nGenerating predictions...")
    X, df_with_features = prepare_data_for_prediction(df, feature_names, injuries_dict, form_dict)
    
    predictions = model.predict(X)
    
    df_with_features["predicted_xGoalsPercentage_5on5"] = predictions
    df_with_features["prediction_error"] = (
        df_with_features["predicted_xGoalsPercentage_5on5"] - 
        df_with_features[TARGET_COLUMN]
    )
    df_with_features["absolute_error"] = df_with_features["prediction_error"].abs()
    
    output_cols = [
        'team', 'season', TARGET_COLUMN, 'predicted_xGoalsPercentage_5on5',
        'prediction_error', 'absolute_error', 'injury_penalty', 'recent_form'
    ]
    output_df = df_with_features[output_cols]
    
    output_df.to_csv("predictions.csv", index=False)
    
    logger.info(f"\n✅ Saved {len(output_df)} predictions to predictions.csv")
    logger.info(f"Mean Absolute Error: {output_df['absolute_error'].mean():.4f}")
    
    print(output_df.head(10).to_string())

if __name__ == "__main__":
    main()
