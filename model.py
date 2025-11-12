import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 IMDb RATING PREDICTION MODEL")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/5] Loading processed data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"   Train shape: {train_df.shape}")
print(f"   Test shape: {test_df.shape}")

# Split features and target
X_train = train_df.drop('avg_vote', axis=1)
y_train = train_df['avg_vote']
X_test = test_df.drop('avg_vote', axis=1)
y_test = test_df['avg_vote']

print(f"   Features: {X_train.shape[1]}")
print(f"   Train samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Target range: [{y_train.min():.1f}, {y_train.max():.1f}]")

# ============================================================================
# STEP 2: Train Models
# ============================================================================
print("\n[2/5] Training models...")
print("="*80)

models = {}
predictions = {}

# Model 1: Random Forest
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_model.predict(X_test)
print("   ✓ Random Forest trained")

# Model 2: XGBoost
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method='gpu_hist',  # Use GPU
    gpu_id=0
)
xgb_model.fit(X_train, y_train, verbose=50)
models['XGBoost'] = xgb_model
predictions['XGBoost'] = xgb_model.predict(X_test)
print("   ✓ XGBoost trained")

# Model 3: LightGBM
print("\n⚡ Training LightGBM...")
try:
    # Try GPU first
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        device='cuda',
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    print("   ✓ LightGBM trained (GPU)")
except Exception as e:
    # Fall back to CPU
    print(f"   ⚠ GPU not available for LightGBM, using CPU instead")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    print("   ✓ LightGBM trained (CPU)")

models['LightGBM'] = lgb_model
predictions['LightGBM'] = lgb_model.predict(X_test)

# ============================================================================
# STEP 3: Evaluate Models
# ============================================================================
print("\n[3/5] Evaluating models...")
print("="*80)

results = []
for name, y_pred in predictions.items():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })

    print(f"\n{name}:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")

results_df = pd.DataFrame(results)

# Find best model
best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
best_predictions = predictions[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print("="*80)

# ============================================================================
# STEP 4: Feature Importance (Best Model)
# ============================================================================
print(f"\n[4/5] Analyzing feature importance ({best_model_name})...")

best_model = models[best_model_name]
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("\n[5/5] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Model Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(results_df))
bars = ax1.bar(x_pos, results_df['RMSE'], color=['#3498db', '#e74c3c', '#2ecc71'])
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Model'], rotation=0)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontweight='bold')

# Plot 2: Predictions vs Actual (Best Model)
ax2 = axes[0, 1]
scatter = ax2.scatter(y_test, best_predictions, alpha=0.5, c=y_test, cmap='viridis', s=10)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Rating', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
ax2.set_title(f'Predictions vs Actual - {best_model_name}', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Actual Rating')

# Plot 3: Residual Distribution
ax3 = axes[1, 0]
residuals = y_test - best_predictions
ax3.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax3.set_xlabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title(f'Residual Distribution - {best_model_name}', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Top 15 Feature Importances
ax4 = axes[1, 1]
top_features = feature_importance.head(15)
y_pos = np.arange(len(top_features))
ax4.barh(y_pos, top_features['importance'], color='coral')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(top_features['feature'], fontsize=9)
ax4.invert_yaxis()
ax4.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax4.set_title(f'Top 15 Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: model_evaluation.png")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n💾 Saving results...")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': best_predictions,
    'residual': y_test.values - best_predictions
})
predictions_df.to_csv('predictions.csv', index=False)
print("   ✓ Saved: predictions.csv")

# Save model comparison
results_df.to_csv('model_comparison.csv', index=False)
print("   ✓ Saved: model_comparison.csv")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("   ✓ Saved: feature_importance.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ MODEL TRAINING AND EVALUATION COMPLETE!")
print("="*80)

print(f"\n📊 Results Summary:")
print(f"   Best Model: {best_model_name}")
print(f"   RMSE: {results_df.loc[results_df['Model'] == best_model_name, 'RMSE'].values[0]:.4f}")
print(f"   MAE:  {results_df.loc[results_df['Model'] == best_model_name, 'MAE'].values[0]:.4f}")
print(f"   R²:   {results_df.loc[results_df['Model'] == best_model_name, 'R²'].values[0]:.4f}")

print(f"\n🎯 Interpretation:")
rmse_val = results_df.loc[results_df['Model'] == best_model_name, 'RMSE'].values[0]
mae_val = results_df.loc[results_df['Model'] == best_model_name, 'MAE'].values[0]
print(f"   On average, predictions are off by {mae_val:.4f} rating points")
print(f"   95% of predictions are within {2*rmse_val:.4f} rating points")

print(f"\n📁 Files Created:")
print(f"   • model_evaluation.png - Visual analysis")
print(f"   • predictions.csv - Test set predictions")
print(f"   • model_comparison.csv - Model performance comparison")
print(f"   • feature_importance.csv - Feature importance rankings")

print("\n🚀 Models trained and ready for use!")
print("="*80)
