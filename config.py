"""
Configuration settings for Moose Picks
"""

DATA_DIR = "data"
MODELS_DIR = "models"

CSV_FILES = ["teams_2024.csv", "teams_2025.csv"]

TARGET_COLUMN = "xGoalsPercentage_5on5"

MODEL_PATH = f"{MODELS_DIR}/m3_local_model.pkl"
PREDICTION_OUTPUT = "predictions.csv"

SITUATIONS = ["5on5", "5on4", "4on5", "4on4"]

BASE_STATS = [
    "xGoalsPercentage", "corsiPercentage", "fenwickPercentage",
    "goalsFor", "goalsAgainst", "shotsOnGoalFor", "shotsOnGoalAgainst",
    "highDangerGoalsFor", "highDangerGoalsAgainst", "reboundsFor", "reboundsAgainst"
]

ENGINEERED_FEATURES = [
    "goal_diff_5on5", "shot_diff_5on5", "high_danger_diff_5on5",
    "shooting_pct_5on5", "save_pct_5on5",
    "injury_penalty",
    "recent_form",
    "h2h_vs_opponent"
]
