"""
OffSide 2026 v3 — CLEAN Solution (NO TARGET LEAKAGE)
Fix: Remove ALL game-level and player-history features derived from scored_flag
     that aren't cross-validated. Keep ONLY CV-protected target encoding.
     Add: CatBoost, better xG imputation, non-leaky interaction features.
"""
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings, gc, time

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

print("=" * 70)
print("  OffSide 2026 v3 — CLEAN (No Target Leakage)")
print("  LightGBM + XGBoost + CatBoost Ensemble")
print("=" * 70)

# ============================================================
# 1. DATA LOADING
# ============================================================
t0 = time.time()
print("\n[1/8] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"  Train: {train.shape}  Test: {test.shape}  ({time.time()-t0:.1f}s)")
train['scored_flag'] = train['scored_flag'].astype(int)

# ============================================================
# 2. FEATURE ENGINEERING (NO LEAKAGE)
# ============================================================
print("\n[2/8] Feature engineering (leak-free)...")
t1 = time.time()
n_train = len(train)
n_test = len(test)
y_train = train['scored_flag'].values
appearance_ids_test = test['appearance_id'].values

# Extract IDs
for df_tmp in [train, test]:
    df_tmp['player_id'] = df_tmp['appearance_id'].str.split('_').str[0].astype(int)
    df_tmp['game_id'] = df_tmp['appearance_id'].str.split('_').str[1].astype(int)
    df_tmp['date'] = pd.to_datetime(df_tmp['date'])

test['scored_flag'] = -1
df = pd.concat([train, test], axis=0, ignore_index=True)
train_mask = df.index < n_train

# --- Match context (from available columns, NOT from scored_flag) ---
df['is_home'] = (df['home_away'] == 'HOME').astype(int)
df['team_goals'] = np.where(df['is_home']==1, df['home_club_goals'], df['away_club_goals'])
df['opponent_goals'] = np.where(df['is_home']==1, df['away_club_goals'], df['home_club_goals'])
df['total_match_goals'] = df['home_club_goals'] + df['away_club_goals']
df['goal_margin'] = df['team_goals'] - df['opponent_goals']
df['player_club_name'] = np.where(df['is_home']==1, df['home_club_name'], df['away_club_name'])
df['opponent_club_name'] = np.where(df['is_home']==1, df['away_club_name'], df['home_club_name'])
df['team_goal_share'] = df['team_goals'] / (df['total_match_goals'] + 1e-8)
df['is_blowout'] = (df['goal_diff_abs'] >= 3).astype(int)
df['is_close_match'] = (df['goal_diff_abs'] <= 1).astype(int)
df['team_won'] = (df['goal_margin'] > 0).astype(int)
df['team_drew'] = (df['goal_margin'] == 0).astype(int)
df['team_lost'] = (df['goal_margin'] < 0).astype(int)

# --- Date features ---
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_year'] = df['date'].dt.dayofyear
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['quarter'] = df['date'].dt.quarter
df['season_half'] = (df['month'] >= 7).astype(int)
df['year'] = df['date'].dt.year

# --- Player profile ---
df['age_squared'] = df['age'] ** 2
df['is_young_talent'] = ((df['age'] >= 18) & (df['age'] <= 22)).astype(int)
df['is_prime'] = ((df['age'] >= 25) & (df['age'] <= 29)).astype(int)
df['foot'] = df['foot'].fillna('Unknown')

# --- Smart xG imputation (position-based medians instead of 0) ---
xg_cols = ['avg_xG', 'avg_xA', 'avg_shots', 'avg_key_passes', 'avg_xGChain', 'avg_xGBuildup', 'avg_npxG']
for col in xg_cols:
    pos_medians = df[train_mask].groupby('position')[col].median()
    for pos, med in pos_medians.items():
        mask = df[col].isna() & (df['position'] == pos)
        df.loc[mask, col] = med

# --- xG interaction features (CRITICAL — all from non-target columns) ---
df['xG_x_minutes'] = df['avg_xG'] * df['minutes_ratio']
df['npxG_x_minutes'] = df['avg_npxG'] * df['minutes_ratio']
df['shots_x_minutes'] = df['avg_shots'] * df['minutes_ratio']
df['key_passes_x_minutes'] = df['avg_key_passes'] * df['minutes_ratio']
df['xG_x_attacker'] = df['avg_xG'] * df['is_attacker'].astype(int)
df['xG_x_midfielder'] = df['avg_xG'] * df['is_midfielder'].astype(int)
df['xG_x_defender'] = df['avg_xG'] * df['is_defender'].astype(int)
df['xG_per_shot'] = df['avg_xG'] / (df['avg_shots'] + 1e-8)
df['npxG_per_shot'] = df['avg_npxG'] / (df['avg_shots'] + 1e-8)
df['xG_plus_xA'] = df['avg_xG'] + df['avg_xA']
df['xG_minus_npxG'] = df['avg_xG'] - df['avg_npxG']
df['chain_buildup_ratio'] = df['avg_xGChain'] / (df['avg_xGBuildup'] + 1e-8)
df['chain_minus_buildup'] = df['avg_xGChain'] - df['avg_xGBuildup']

# --- xG × match context (non-leaky) ---
df['xG_x_team_goals'] = df['avg_xG'] * df['team_goals']
df['xG_x_total_goals'] = df['avg_xG'] * df['total_match_goals']
df['npxG_x_team_goals'] = df['avg_npxG'] * df['team_goals']
df['shots_x_team_goals'] = df['avg_shots'] * df['team_goals']
df['xG_x_is_home'] = df['avg_xG'] * df['is_home']
df['xG_x_goal_margin'] = df['avg_xG'] * df['goal_margin']

# --- Market interactions ---
df['market_x_xG'] = df['log_market_value'].fillna(0) * df['avg_xG']
df['value_per_age'] = df['market_value_before_match'].fillna(0) / (df['age'].fillna(25) + 1)
df['market_x_minutes'] = df['log_market_value'].fillna(0) * df['minutes_ratio']
df['market_x_attacker'] = df['log_market_value'].fillna(0) * df['is_attacker'].astype(int)

# --- International ---
df['intl_scoring_eff'] = df['international_goals'].fillna(0) / (df['international_caps'].fillna(0) + 1)
df['caps_x_goal_rate'] = df['international_caps'].fillna(0) * df['goal_per_cap'].fillna(0)
df['intl_goals_x_xG'] = df['international_goals'].fillna(0) * df['avg_xG']

# --- Minutes features ---
df['played_very_little'] = (df['minutes_played'] < 15).astype(int)
df['played_full'] = (df['minutes_played'] >= 85).astype(int)
df['minutes_bucket'] = pd.cut(df['minutes_played'], bins=[-1,10,30,60,80,90,200], labels=[0,1,2,3,4,5]).astype(float)

# --- Composite scoring threat (from non-target features) ---
df['scoring_threat'] = (df['avg_xG']*3 + df['avg_npxG']*2 +
                        df['avg_shots']*0.1 + df['is_attacker'].astype(int)*0.5 +
                        df['finisher_flag'].astype(int)*0.3)
df['scoring_threat_x_minutes'] = df['scoring_threat'] * df['minutes_ratio']
df['scoring_threat_x_team_goals'] = df['scoring_threat'] * df['team_goals']

# --- Frequency encoding (non-leaky — count-based, not target-based) ---
player_freq = df['player_id'].value_counts().to_dict()
df['player_frequency'] = df['player_id'].map(player_freq)
team_freq = df['player_club_name'].value_counts().to_dict()
df['team_frequency'] = df['player_club_name'].map(team_freq)
opp_freq = df['opponent_club_name'].value_counts().to_dict()
df['opponent_frequency'] = df['opponent_club_name'].map(opp_freq)

# --- Player avg stats (from NON-TARGET columns only) ---
# These are safe because they don't use scored_flag
player_xg_stats = df[train_mask].groupby('player_id').agg(
    player_avg_xG=('avg_xG', 'mean'),
    player_avg_minutes=('minutes_played', 'mean'),
    player_avg_shots=('avg_shots', 'mean'),
    player_n_apps=('scored_flag', 'count'),  # count is safe (doesn't use target values)
    player_avg_team_goals=('team_goals', 'mean'),
).reset_index()
df = df.merge(player_xg_stats, on='player_id', how='left')

# Player xG × their average (consistency signal)
df['xG_vs_player_avg'] = df['avg_xG'] - df['player_avg_xG'].fillna(0)
df['minutes_vs_player_avg'] = df['minutes_played'] - df['player_avg_minutes'].fillna(df['minutes_played'])

# --- Team avg stats (from NON-TARGET columns) ---
team_goal_stats = df[train_mask].groupby('player_club_name').agg(
    team_avg_team_goals=('team_goals', 'mean'),
    team_avg_opp_goals=('opponent_goals', 'mean'),
).reset_index()
df = df.merge(team_goal_stats, on='player_club_name', how='left')

opp_goal_stats = df[train_mask].groupby('opponent_club_name').agg(
    opp_avg_goals_conceded=('team_goals', 'mean'),
).reset_index()
opp_goal_stats.columns = ['opponent_club_name', 'opp_avg_goals_conceded']
df = df.merge(opp_goal_stats, on='opponent_club_name', how='left')

print(f"  Feature engineering done. Shape: {df.shape}  ({time.time()-t1:.1f}s)")

# ============================================================
# 3. TARGET ENCODING (CV-PROTECTED — SAFE)
# ============================================================
print("\n[3/8] Target encoding (cross-validated, leak-free)...")
t2 = time.time()

def target_encode_cv(df, col, n_folds=5, smoothing=50, seed=42):
    train_data = df[train_mask].copy()
    test_data = df[~train_mask].copy()
    global_mean = train_data['scored_flag'].mean()
    new_col = f'{col}_te'
    df[new_col] = np.nan
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_data, train_data['scored_flag']):
        fold_train = train_data.iloc[tr_idx]
        agg = fold_train.groupby(col)['scored_flag'].agg(['mean','count'])
        smoothed = (agg['count']*agg['mean'] + smoothing*global_mean) / (agg['count'] + smoothing)
        val_enc = train_data.iloc[val_idx][col].map(smoothed)
        df.loc[train_data.index[val_idx], new_col] = val_enc.values
    agg = train_data.groupby(col)['scored_flag'].agg(['mean','count'])
    smoothed = (agg['count']*agg['mean'] + smoothing*global_mean) / (agg['count'] + smoothing)
    df.loc[test_data.index, new_col] = test_data[col].map(smoothed).values
    df[new_col].fillna(global_mean, inplace=True)
    return df

# Multiple smoothing levels for player_id
for smooth in [10, 50, 200]:
    df = target_encode_cv(df, 'player_id', smoothing=smooth)
    df.rename(columns={'player_id_te': f'player_id_te_s{smooth}'}, inplace=True)
    print(f"  TE: player_id (smoothing={smooth})")

for col in ['sub_position', 'player_club_name', 'opponent_club_name',
            'stadium', 'referee', 'country_of_citizenship', 'name_x']:
    if col in df.columns:
        df = target_encode_cv(df, col, smoothing=50)
        print(f"  TE: {col}")

# NEW: CV-protected player×competition and player×home_away encoding
df['player_comp'] = df['player_id'].astype(str) + '_' + df['name_x'].astype(str)
df = target_encode_cv(df, 'player_comp', smoothing=20)
print("  TE: player_comp (player×competition)")

df['player_ha'] = df['player_id'].astype(str) + '_' + df['home_away'].astype(str)
df = target_encode_cv(df, 'player_ha', smoothing=20)
print("  TE: player_ha (player×home_away)")

df['subpos_comp'] = df['sub_position'].astype(str) + '_' + df['name_x'].astype(str)
df = target_encode_cv(df, 'subpos_comp', smoothing=30)
print("  TE: subpos_comp (sub_position×competition)")

print(f"  Target encoding done. ({time.time()-t2:.1f}s)")

# Drop temp columns used for composite TE
df.drop(columns=['player_comp', 'player_ha', 'subpos_comp'], inplace=True, errors='ignore')

# ============================================================
# 4. LABEL ENCODING
# ============================================================
print("\n[4/8] Label encoding categoricals...")
cat_cols = ['home_away','position','sub_position','foot','competition_type',
            'confederation','market_value_tier','age_bucket','country_name',
            'home_club_name','away_club_name','player_club_name','opponent_club_name',
            'name_x','stadium','referee','country_of_citizenship']
cat_cols = [c for c in cat_cols if c in df.columns]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna('__MISSING__').astype(str)
    df[col] = le.fit_transform(df[col])

# ============================================================
# 5. PREPARE FEATURES
# ============================================================
print("\n[5/8] Preparing feature matrix...")
drop_cols = ['appearance_id','date','scored_flag','name_y','home_club_id','away_club_id',
             'player_id','game_id']
drop_cols = [c for c in drop_cols if c in df.columns]

bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    df[col] = df[col].astype(int)

feature_cols = [c for c in df.columns if c not in drop_cols]

X_train = df.loc[df.index < n_train, feature_cols].copy()
X_test = df.loc[df.index >= n_train, feature_cols].copy()
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        bool_map = {'True': 1, 'False': 0, 'true': 1, 'false': 0, True: 1, False: 0}
        X_train[col] = X_train[col].map(bool_map).fillna(-1).astype(float)
        X_test[col] = X_test[col].map(bool_map).fillna(-1).astype(float)
    elif X_train[col].dtype == 'category':
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)

print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}  Features: {len(feature_cols)}")
del df; gc.collect()

# ============================================================
# 6. MODEL TRAINING
# ============================================================
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos = neg_count / pos_count
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# --- LightGBM ---
print(f"\n[6/8] Training models...")
print(f"\n{'='*55}\n  LightGBM — {N_FOLDS}-Fold CV\n{'='*55}")

lgb_params = {
    'objective': 'binary', 'metric': 'average_precision', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'num_leaves': 127, 'max_depth': -1,
    'min_child_samples': 80, 'feature_fraction': 0.7, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos, 'verbose': -1, 'n_jobs': -1, 'random_state': SEED,
}

lgb_oof = np.zeros(n_train)
lgb_test = np.zeros(n_test)
lgb_scores = []
lgb_models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    t_fold = time.time()
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    val_preds = model.predict(X_val)
    lgb_oof[val_idx] = val_preds
    lgb_test += model.predict(X_test) / N_FOLDS
    ap = average_precision_score(y_val, val_preds)
    lgb_scores.append(ap)
    lgb_models.append(model)
    print(f"  Fold {fold+1} AP: {ap:.6f}  ({time.time()-t_fold:.0f}s)")

lgb_oof_ap = average_precision_score(y_train, lgb_oof)
print(f"\n  LightGBM OOF AP: {lgb_oof_ap:.6f}  (Mean: {np.mean(lgb_scores):.6f} +/- {np.std(lgb_scores):.6f})")

# --- XGBoost ---
print(f"\n{'='*55}\n  XGBoost — {N_FOLDS}-Fold CV\n{'='*55}")

xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'learning_rate': 0.03, 'max_depth': 8, 'min_child_weight': 80,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos, 'tree_method': 'hist',
    'random_state': SEED, 'n_jobs': -1,
}

xgb_oof = np.zeros(n_train)
xgb_test = np.zeros(n_test)
xgb_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    t_fold = time.time()
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                      evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=500)
    val_preds = model.predict(dval)
    xgb_oof[val_idx] = val_preds
    xgb_test += model.predict(xgb.DMatrix(X_test)) / N_FOLDS
    ap = average_precision_score(y_val, val_preds)
    xgb_scores.append(ap)
    print(f"  Fold {fold+1} AP: {ap:.6f}  ({time.time()-t_fold:.0f}s)")

xgb_oof_ap = average_precision_score(y_train, xgb_oof)
print(f"\n  XGBoost OOF AP: {xgb_oof_ap:.6f}  (Mean: {np.mean(xgb_scores):.6f} +/- {np.std(xgb_scores):.6f})")

# --- CatBoost (fast) ---
print(f"\n{'='*55}\n  CatBoost (fast) — {N_FOLDS}-Fold CV\n{'='*55}")

cb_oof = np.zeros(n_train)
cb_test = np.zeros(n_test)
cb_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    t_fold = time.time()
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        min_child_samples=80, subsample=0.8, colsample_bylevel=0.7,
        random_seed=SEED, auto_class_weights='Balanced', eval_metric='PRAUC',
        early_stopping_rounds=100, verbose=500, task_type='CPU')
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    val_preds = model.predict_proba(X_val)[:, 1]
    cb_oof[val_idx] = val_preds
    cb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS
    ap = average_precision_score(y_val, val_preds)
    cb_scores.append(ap)
    print(f"  Fold {fold+1} AP: {ap:.6f}  ({time.time()-t_fold:.0f}s)")

cb_oof_ap = average_precision_score(y_train, cb_oof)
print(f"\n  CatBoost OOF AP: {cb_oof_ap:.6f}  (Mean: {np.mean(cb_scores):.6f} +/- {np.std(cb_scores):.6f})")

# ============================================================
# 7. ENSEMBLE
# ============================================================
print(f"\n[7/8] Ensemble optimization...")

best_ap = 0
best_w = (0.33, 0.33, 0.34)

for w1 in np.arange(0.05, 0.9, 0.05):
    for w2 in np.arange(0.05, 0.9 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05: continue
        blend = w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof
        ap = average_precision_score(y_train, blend)
        if ap > best_ap:
            best_ap = ap
            best_w = (w1, w2, w3)

print(f"\n  Best Weights: LGB={best_w[0]:.2f}  XGB={best_w[1]:.2f}  CB={best_w[2]:.2f}")
print(f"\n  --- FINAL RESULTS ---")
print(f"  LightGBM OOF AP:  {lgb_oof_ap:.6f}")
print(f"  XGBoost  OOF AP:  {xgb_oof_ap:.6f}")
print(f"  CatBoost OOF AP:  {cb_oof_ap:.6f}")
print(f"  ENSEMBLE OOF AP:  {best_ap:.6f}")
print(f"  v1 OOF AP was:    0.478443 (public LB: ~0.50)")

final_preds = best_w[0]*lgb_test + best_w[1]*xgb_test + best_w[2]*cb_test

# ============================================================
# 8. PLOTS & SUBMISSION
# ============================================================
print(f"\n[8/8] Saving plots and submission...")

# Model evaluation plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble']
model_aps = [lgb_oof_ap, xgb_oof_ap, cb_oof_ap, best_ap]
bars_colors = ['#4CAF50', '#2196F3', '#FF9800', '#FF5722']
bars = axes[0].bar(model_names, model_aps, color=bars_colors, edgecolor='white')
axes[0].set_title('Model Comparison (OOF AP)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Average Precision')
for bar, ap_val in zip(bars, model_aps):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f'{ap_val:.5f}', ha='center', fontsize=10, fontweight='bold')
axes[0].set_ylim(min(model_aps)-0.02, max(model_aps)+0.02)

oof_ens = best_w[0]*lgb_oof + best_w[1]*xgb_oof + best_w[2]*cb_oof
prec, rec, _ = precision_recall_curve(y_train, oof_ens)
axes[1].plot(rec, prec, color='#FF5722', linewidth=2, label=f'Ensemble (AP={best_ap:.4f})')
axes[1].fill_between(rec, prec, alpha=0.1, color='#FF5722')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold'); axes[1].legend()

axes[2].hist(final_preds, bins=100, color='#2196F3', alpha=0.7, edgecolor='white')
axes[2].axvline(x=final_preds.mean(), color='red', linestyle='--', label=f'Mean: {final_preds.mean():.4f}')
axes[2].set_title('Test Prediction Distribution', fontsize=13, fontweight='bold'); axes[2].legend()

for ax in axes: ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig('part2/model_evaluation_v3.png', dpi=150, bbox_inches='tight'); plt.close()

# Feature importance
fig, ax = plt.subplots(figsize=(12, 12))
imp = pd.DataFrame({'f': feature_cols, 'imp': lgb_models[0].feature_importance('gain')}).sort_values('imp', ascending=False).head(30)
ax.barh(imp['f'].values[::-1], imp['imp'].values[::-1], color=sns.color_palette('viridis', 30))
ax.set_title('Top 30 Features (LightGBM Gain) — v3', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('part2/feature_importance_v3.png', dpi=150, bbox_inches='tight'); plt.close()

print("\n  Top 20 Features:")
for _, row in imp.head(20).iterrows():
    print(f"    {row['f']:50s} {row['imp']:,.0f}")

# Save submission
submission = pd.DataFrame({'appearance_id': appearance_ids_test, 'scored_flag': final_preds})
submission['scored_flag'] = submission['scored_flag'].clip(0.0001, 0.9999)
submission.to_csv('part2/solution.csv', index=False)

total_time = time.time() - t0
print(f"\n{'='*70}")
print(f"  DONE! Total time: {total_time/60:.1f} minutes")
print(f"  Saved: part2/solution.csv ({len(submission):,} predictions)")
print(f"  Mean prediction: {submission['scored_flag'].mean():.6f}")
print(f"  Ensemble OOF AP: {best_ap:.6f}")
print(f"{'='*70}")
