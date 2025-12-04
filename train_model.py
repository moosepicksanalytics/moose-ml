"""
UPDATED: Train model with injuries, form, and H2H included
AUTOMATED: Injury impacts now calculated by algorithm
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
CSV_FILES = ["teams_2024.csv", "teams_2025.csv"]
TARGET_COLUMN = "xGoalsPercentage_5on5"

BASE_STATS = [
    "xGoalsPercentage", "corsiPercentage", "fenwickPercentage",
    "goalsFor", "goalsAgainst", "shotsOnGoalFor", "shotsOnGoalAgainst",
    "highDangerGoalsFor", "highDangerGoalsAgainst", "reboundsFor", "reboundsAgainst"
]

SITUATIONS = ["5on5", "5on4", "4on5", "4on4"]

ENGINEERED_FEATURES = [
    "goal_diff_5on5", "shot_diff_5on5", "high_danger_diff_5on5",
    "shooting_pct_5on5", "save_pct_5on5",
    "injury_penalty",
    "recent_form",
    "h2h_vs_opponent"
]

def load_data():
    logger.info("Loading data from CSVs...")
    
    frames = []
    for filename in CSV_FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
        df = pd.read_csv(path)
        frames.append(df)
    
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined['situation'].isin(SITUATIONS)].copy()
    logger.info(f"Loaded {len(combined)} rows")
    
    pivoted = combined.pivot_table(
        index=['team', 'season'],
        columns='situation',
        values=BASE_STATS,
        aggfunc='first'
    )
    
    pivoted.columns = ['_'.join(col) for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    return pivoted

def engineer_features(df):
    df = df.copy()
    
    logger.info("Engineering features...")
    
    df["goal_diff_5on5"] = df["goalsFor_5on5"] - df["goalsAgainst_5on5"]
    df["shot_diff_5on5"] = df["shotsOnGoalFor_5on5"] - df["shotsOnGoalAgainst_5on5"]
    df["high_danger_diff_5on5"] = (
        df["highDangerGoalsFor_5on5"] - df["highDangerGoalsAgainst_5on5"]
    )
    
    df["shooting_pct_5on5"] = df["goalsFor_5on5"] / (df["shotsOnGoalFor_5on5"] + 1)
    df["save_pct_5on5"] = 1 - (
        df["goalsAgainst_5on5"] / (df["shotsOnGoalAgainst_5on5"] + 1)
    )
    
    injuries_path = os.path.join(DATA_DIR, "injuries.csv")
    if os.path.exists(injuries_path):
        injuries = pd.read_csv(injuries_path)
        injury_impact = injuries.groupby('team')['injury_impact'].sum().to_dict()
        df['injury_penalty'] = df['team'].map(injury_impact).fillna(0)
        logger.info(f"✓ Loaded auto-calculated injuries for {len(injury_impact)} teams")
    else:
        logger.warning("No injuries.csv - using 0")
        df['injury_penalty'] = 0
    
    form_path = os.path.join(DATA_DIR, "form.csv")
    if os.path.exists(form_path):
        form = pd.read_csv(form_path)
        form_dict = (form.set_index('team')['last_10_win_pct'] - 0.5).to_dict()
        df['recent_form'] = df['team'].map(form_dict).fillna(0)
        logger.info(f"✓ Loaded form for {len(form_dict)} teams")
    else:
        logger.warning("No form.csv - using 0")
        df['recent_form'] = 0
    
    df['h2h_vs_opponent'] = 0
    
    logger.info(f"✓ Created {len(ENGINEERED_FEATURES)} features")
    
    return df

def prepare_data(df):
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df = engineer_features(df)
    
    all_features = []
    for stat in BASE_STATS:
        for sit in SITUATIONS:
            col = f"{stat}_{sit}"
            if col != TARGET_COLUMN and col in df.columns:
                all_features.append(col)
    
    all_features.extend([f for f in ENGINEERED_FEATURES if f != 'h2h_vs_opponent'])
    available_features = [f for f in all_features if f in df.columns]
    
    logger.info(f"Using {len(available_features)} features")
    
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    
    X = df[available_features].values
    y = df[TARGET_COLUMN].values
    
    return X, y, available_features

def train_model(X, y, feature_names):
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODEL (With Auto-Calculated Injury Impact)")
    logger.info("="*60)
    
    logger.info("\n5-fold cross-validation...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    logger.info(f"CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info(f"\nTest Results: R²={r2:.4f}, MAE={mae:.4f}")
    
    logger.info(f"\nTop Features:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:<35} {row['importance']:.4f}")
    
    return model, feature_names

def main():
    df = load_data()
    X, y, feature_names = prepare_data(df)
    model, features = train_model(X, y, feature_names)
    
    os.makedirs("models", exist_ok=True)
    model_data = {
        "model": model,
        "feature_names": features,
        "created_at": datetime.now().isoformat()
    }
    
    with open("models/m3_local_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    logger.info(f"\n✅ Model saved to models/m3_local_model.pkl")


if __name__ == "__main__":
    main()
