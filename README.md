# Moose Picks ML

Fully automated sports analytics and betting prediction system for NHL.

## Features

- Auto-downloads hockey stats daily
- Auto-estimates player injury impacts
- Calculates team form and head-to-head records
- Trains advanced ML model daily
- Generates betting predictions
- Updates automatically every morning at 6 AM

## Setup

```bash
pip install -r requirements.txt
python train_model.py
python predict_from_model.py
```

## Structure

- `data/` – CSV data files (teams stats, game history, injuries, form, H2H)
- `models/` – Trained ML model files
- `scripts/` – Automation scripts
- `train_model.py` – Model training
- `predict_from_model.py` – Prediction generation
- `daily_pipeline.py` – Complete automated pipeline

## Daily Workflow

Pipeline runs automatically every day at 6 AM and:
1. Downloads latest stats
2. Creates/updates game history
3. Auto-estimates injuries
4. Calculates form and H2H
5. Trains model
6. Generates predictions
7. Updates predictions.csv

## Output

Results saved to `predictions.csv` with columns:
- `team` – NHL team name
- `season` – Season year
- `xGoalsPercentage_5on5` – Actual expected goals %
- `predicted_xGoalsPercentage_5on5` – Model prediction
- `prediction_error` – Difference
- `injury_penalty` – Auto-estimated injury impact
- `recent_form` – Team's last 10 games win %
