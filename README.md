# OffSide 2026 — Datathon MUJ | Round 1 Solution

## 🏆 Results
- **Public Leaderboard Score: 0.55025 AP**
- **Leaderboard Position: 4th**
- **Metric: Average Precision (AP)**

## 📋 Problem Statement
Predict the probability that a football player scores (`scored_flag`) in a given match appearance, using player statistics, match context, and historical performance data.

- **Train set:** 1,319,813 appearances × 64 features
- **Test set:** 565,635 appearances × 63 features
- **Target:** `scored_flag` (binary: 0 or 1)
- **Evaluation Metric:** Average Precision (AP)

## 🧠 Approach

### Feature Engineering (Leak-Free)
All features are derived without target leakage — no direct use of `scored_flag` outside of cross-validated target encoding.

1. **Match Context Features:**
   - Team goals, opponent goals, goal margin, goal share
   - Home/away indicator, match result (win/draw/loss)
   - Blowout and close match flags

2. **Player Profile Features:**
   - Age interactions (squared, talent/prime buckets)
   - Market value interactions (value × xG, value × minutes)
   - International scoring efficiency

3. **Smart xG Imputation:**
   - Position-based median imputation for missing xG stats (~48% missing)
   - Replaces naive zero-fill with informed estimates

4. **xG Interaction Features:**
   - xG × minutes_ratio, xG × team_goals, xG × goal_margin
   - npxG × minutes, shots × team_goals
   - xG per shot, chain-buildup ratio

5. **Composite Scoring Threat:**
   - Weighted combination of xG, npxG, shots, position, finisher flag
   - Interactions with minutes and team goals

6. **CV-Protected Target Encoding (13 encodings):**
   - Player ID (3 smoothing levels: 10, 50, 200)
   - Sub-position, club, opponent, stadium, referee, nationality, competition
   - Composite: player×competition, player×home_away, sub_position×competition

7. **Non-Target Player Statistics:**
   - Player average xG, shots, minutes (from non-target columns only)
   - Team and opponent average goals (non-target derived)
   - xG deviation from player average

8. **Frequency Encoding:**
   - Player, team, and opponent appearance counts

### Model Architecture

**3-Model Ensemble: LightGBM + XGBoost + CatBoost**

| Model | Learning Rate | OOF AP | Ensemble Weight |
|-------|--------------|--------|-----------------|
| LightGBM | 0.03 | 0.5104 | 25% |
| XGBoost | 0.03 | 0.5134 | 60% |
| CatBoost | 0.05 | 0.5073 | 15% |
| **Ensemble** | — | **0.5145** | — |

- **Cross-Validation:** 5-Fold Stratified K-Fold (seed=42)
- **Early Stopping:** 100 rounds patience
- **Class Imbalance:** Handled via `scale_pos_weight` (LGB/XGB) and `auto_class_weights='Balanced'` (CatBoost)
- **Ensemble Optimization:** Grid search over model weights to maximize OOF AP

### Key Design Decisions
- **No AutoML** — all models hand-tuned per competition rules
- **No target leakage** — all target-derived features are cross-validated
- **Position-based xG imputation** — critical for the 48% missing xG data
- **Multiple target encoding smoothing levels** — captures both high-frequency and rare player patterns

## 📂 Files

| File | Description |
|------|-------------|
| `solution.py` | Main solution script (produces `solution.csv`) |
| `solution.csv` | Final submission file (565,635 predictions) |
| `model_evaluation.png` | Model comparison and PR curve |
| `feature_importance.png` | Top 30 feature importances |

## 🔧 Requirements

```
python >= 3.10
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
matplotlib
seaborn
```

## ▶️ How to Run

```bash
cd <project_root>
python part2/solution.py
```

The script reads `train.csv` and `test.csv` from the project root and saves `solution.csv` to `part2/`.

**Runtime:** ~97 minutes on a standard CPU machine.

## 📊 Top 20 Most Important Features

| Rank | Feature | Type |
|------|---------|------|
| 1 | scoring_threat × team_goals | Interaction |
| 2 | player_ha_te | Target Encoding |
| 3 | team_goals | Match Context |
| 4 | shots × team_goals | Interaction |
| 5 | sub_position_te | Target Encoding |
| 6 | subpos_comp_te | Target Encoding |
| 7 | minutes_played | Raw Feature |
| 8 | scoring_threat × minutes | Interaction |
| 9 | xG × minutes | Interaction |
| 10 | player_avg_team_goals | Player Stats |
| 11 | avg_xGChain | Raw Feature |
| 12 | xG × team_goals | Interaction |
| 13 | team_goal_share | Match Context |
| 14 | npxG × team_goals | Interaction |
| 15 | market × minutes | Interaction |
| 16 | attendance | Raw Feature |
| 17 | minutes_vs_player_avg | Player Stats |
| 18 | player_id_te_s10 | Target Encoding |
| 19 | avg_xGBuildup | Raw Feature |
| 20 | value_pct_of_peak | Raw Feature |

## 👤 Mavericks
Datathon MUJ 2026 — Round 1 Submission.
