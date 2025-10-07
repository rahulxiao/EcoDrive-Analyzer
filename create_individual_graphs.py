import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Constants
CONSUMPTION_COL = 'consumption(kWh/100km)'
QUANTITY_COL = 'quantity(kWh)'
TRIP_DISTANCE_COL = 'trip_distance(km)'
AVG_SPEED_COL = 'avg_speed(km/h)'
POWER_COL = 'power(kW)'
ECR_DEVIATION_COL = 'ecr_deviation'

def create_individual_graph_folders():
    """Create folders for individual graphs"""
    folders = [
        'images/individual_graphs/data_exploration',
        'images/individual_graphs/training_analysis',
        'images/individual_graphs/model_performance',
        'images/individual_graphs/data_exploration/dataset_overview',
        'images/individual_graphs/data_exploration/target_distributions',
        'images/individual_graphs/data_exploration/correlations',
        'images/individual_graphs/data_exploration/missing_values',
        'images/individual_graphs/data_exploration/feature_distributions',
        'images/individual_graphs/data_exploration/categorical_distributions',
        'images/individual_graphs/training_analysis/data_distribution',
        'images/individual_graphs/training_analysis/learning_curves',
        'images/individual_graphs/training_analysis/feature_importance',
        'images/individual_graphs/training_analysis/predictions',
        'images/individual_graphs/training_analysis/residuals',
        'images/individual_graphs/model_performance/metrics',
        'images/individual_graphs/model_performance/success_rates',
        'images/individual_graphs/model_performance/radar_charts',
        'images/individual_graphs/model_performance/best_models',
        'images/individual_graphs/model_performance/confusion_matrices'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

def load_csv_with_encoding(file_path):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Loaded {file_path} with {encoding}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {file_path} with any standard encoding")

def load_and_preprocess_data():
    """Load and preprocess the EV dataset"""
    print("Loading datasets...")
    
    # Load datasets with encoding detection
    vw = load_csv_with_encoding("volkswagen_e_golf.csv")
    mitsu = load_csv_with_encoding("mitsubishi_imiev.csv")
    
    # Use Volkswagen columns as primary structure
    all_cols = vw.columns.tolist()
    
    # Add missing columns to Mitsubishi dataset
    for col in all_cols:
        if col not in mitsu.columns:
            mitsu[col] = np.nan
    
    # Ensure both datasets have the same columns
    vw = vw[all_cols]
    mitsu = mitsu[all_cols]
    
    # Combine datasets
    df = pd.concat([vw, mitsu], ignore_index=True)
    
    # Drop fuel_date
    df = df.drop(columns=["fuel_date"], errors="ignore")
    
    # Convert European number format
    numeric_cols = [CONSUMPTION_COL, QUANTITY_COL, ECR_DEVIATION_COL, 
                   TRIP_DISTANCE_COL, AVG_SPEED_COL, POWER_COL]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Handle outliers
    outlier_cols = [CONSUMPTION_COL, QUANTITY_COL, TRIP_DISTANCE_COL, AVG_SPEED_COL]
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    # Feature engineering
    df['energy_efficiency'] = df[QUANTITY_COL] / df[TRIP_DISTANCE_COL]
    df['energy_efficiency'] = df['energy_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    df['speed_category'] = pd.cut(df[AVG_SPEED_COL], 
                                 bins=[0, 30, 60, 90, 200], 
                                 labels=['city', 'urban', 'highway', 'motorway'])
    
    df['distance_category'] = pd.cut(df[TRIP_DISTANCE_COL], 
                                    bins=[0, 10, 50, 100, 1000], 
                                    labels=['short', 'medium', 'long', 'very_long'])
    
    df['power_density'] = df[POWER_COL] / 1500
    df['road_diversity'] = (df['city'] + df['motor_way'] + df['country_roads'])
    df['climate_impact'] = df['A/C'] + df['park_heating']
    
    # Interaction features
    df['speed_distance_interaction'] = df[AVG_SPEED_COL] * df[TRIP_DISTANCE_COL]
    df['power_speed_interaction'] = df[POWER_COL] * df[AVG_SPEED_COL]
    df['distance_efficiency_interaction'] = df[TRIP_DISTANCE_COL] * df['energy_efficiency']
    
    # Polynomial features
    df['speed_squared'] = df[AVG_SPEED_COL] ** 2
    df['distance_squared'] = df[TRIP_DISTANCE_COL] ** 2
    df['power_squared'] = df[POWER_COL] ** 2
    
    # Handle missing values
    target_cols = [CONSUMPTION_COL, QUANTITY_COL, ECR_DEVIATION_COL]
    df = df.dropna(subset=target_cols, how='all')
    
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols_all] = df[numeric_cols_all].fillna(df[numeric_cols_all].median())
    
    categorical_cols = ['tire_type', 'driving_style', 'fuel_type', 'speed_category', 'distance_category']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Unknown')
    
    road_type_cols = ['city', 'motor_way', 'country_roads']
    for col in road_type_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    binary_cols = ['A/C', 'park_heating']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df

def save_confusion_matrix_binned(y_test, y_pred, task_name, num_bins=6):
    """Save a confusion matrix by quantile-binning continuous targets and predictions."""
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    if y_test.size == 0:
        return
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.unique(np.quantile(y_test, quantiles))
    if len(bin_edges) < 3:
        bin_edges = np.linspace(np.nanmin(y_test), np.nanmax(y_test), num_bins + 1)
    y_test_bins = np.digitize(y_test, bin_edges[1:-1], right=True)
    y_pred_bins = np.digitize(y_pred, bin_edges[1:-1], right=True)
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        bin_labels.append(f"[{left:.2f}, {right:.2f})" if i < len(bin_edges) - 2 else f"[{left:.2f}, {right:.2f}]")
    cm = confusion_matrix(y_test_bins, y_pred_bins, labels=list(range(len(bin_labels))))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=bin_labels, yticklabels=bin_labels, ax=ax)
    ax.set_title(f'Binned Confusion Matrix - {task_name.title()}\n(Best Model by R²)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Bin')
    ax.set_ylabel('Actual Bin')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/model_performance/confusion_matrices/17_{task_name}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_dataset_overview_graphs(df):
    """Create individual dataset overview graphs"""
    print("Creating dataset overview graphs...")
    
    # 1. Dataset Statistics
    fig, ax = plt.subplots(figsize=(12, 8))
    
    stats_text = f"""
    📊 DATASET OVERVIEW
    • Total Samples: {len(df):,}
    • Total Features: {len(df.columns)}
    • Missing Values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%)
    • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
    • Data Types: {df.dtypes.value_counts().to_dict()}
    """
    
    ax.text(0.05, 0.5, stats_text, transform=ax.transAxes, fontsize=14, 
            verticalalignment='center', bbox={'boxstyle': 'round,pad=1', 'facecolor': 'lightblue', 'alpha': 0.7})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('📊 Dataset Overview Statistics', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('images/individual_graphs/data_exploration/dataset_overview/01_dataset_statistics.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Target Variables Distribution
    target_cols = [CONSUMPTION_COL, QUANTITY_COL, ECR_DEVIATION_COL]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, (col, color) in enumerate(zip(target_cols, colors)):
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            data = df[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                ax.set_title(f'📈 {col} Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2, 
                          label=f'±1σ: {std_val:.2f}')
                ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2)
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(f'images/individual_graphs/data_exploration/target_distributions/02_{col.replace("(", "").replace(")", "").replace("/", "_")}_distribution.png', 
                            dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
    
    # 3. Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_data = df[numeric_cols].corr()
    
    im = ax.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('🔥 Feature Correlation Matrix', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(corr_data.columns)))
    ax.set_yticks(range(len(corr_data.columns)))
    ax.set_xticklabels(corr_data.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_data.columns)
    
    # Add correlation values
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig('images/individual_graphs/data_exploration/correlations/03_feature_correlation_heatmap.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 4. Missing Values Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        bars = ax.barh(range(len(missing_data)), missing_data.values, 
                       color='lightcoral', alpha=0.8)
        ax.set_yticks(range(len(missing_data)))
        ax.set_yticklabels(missing_data.index)
        ax.set_title('❌ Missing Values by Feature', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Missing Values')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, missing_data.values)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{value}', ha='left', va='center', fontweight='bold')
    else:
        ax.text(0.5, 0.5, '✅ No Missing Values!', ha='center', va='center', 
                fontsize=20, fontweight='bold', color='green')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/individual_graphs/data_exploration/missing_values/04_missing_values_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_additional_exploration_graphs(df):
    """Create additional graphs to match the 12-panel dashboard as individual files.
    This includes numeric feature distributions and categorical feature counts.
    """
    # Numeric feature distributions
    numeric_to_plot = [
        (POWER_COL, '05_power_distribution.png', '⚡ Power (kW) Distribution'),
        (TRIP_DISTANCE_COL, '06_trip_distance_distribution.png', '🛣️ Trip Distance (km) Distribution'),
    ]
    for col, filename, title in numeric_to_plot:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(data, bins=30, alpha=0.7, color='plum', edgecolor='black', linewidth=0.5)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                mean_val = data.mean()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.legend()
                plt.tight_layout()
                plt.savefig(f'images/individual_graphs/data_exploration/feature_distributions/{filename}',
                            dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

    # Categorical feature distributions (top 10 values)
    categorical_to_plot = [
        ('city', '09_city_distribution.png', '🏙️ City Road Usage (Binary)'),
        ('manufacturer', '10_manufacturer_distribution.png', '🏭 Manufacturer Distribution'),
        ('model', '11_model_distribution.png', '🚗 Model Distribution (Top 10)'),
        ('version', '12_version_distribution.png', '🔤 Version Distribution (Top 10)')
    ]
    for col, filename, title in categorical_to_plot:
        if col in df.columns:
            series = df[col].astype(str).fillna('Unknown')
            value_counts = series.value_counts().head(10)
            if len(value_counts) > 0:
                fig, ax = plt.subplots(figsize=(12, 7))
                bars = ax.bar(range(len(value_counts)), value_counts.values, color='teal', alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
                for bar, value in zip(bars, value_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{int(value)}',
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
                plt.tight_layout()
                plt.savefig(f'images/individual_graphs/data_exploration/categorical_distributions/{filename}',
                            dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

def create_training_analysis_graphs(X, y, task_name, models):
    """Create individual training analysis graphs"""
    print(f"Creating training analysis graphs for {task_name}...")
    
    # Remove rows with missing target values
    valid_idx = ~y.isna()
    x_clean = X[valid_idx]
    y_clean = y[valid_idx]
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # 1. Data Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_clean, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    ax.set_title(f'📈 {task_name.title()} Target Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{task_name}')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = y_clean.mean()
    std_val = y_clean.std()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=2, label=f'±1σ: {std_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=2)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/training_analysis/data_distribution/05_{task_name}_target_distribution.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Learning Curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    from sklearn.model_selection import learning_curve
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        model, x_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax.set_title(f'📚 Learning Curves - {task_name.title()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R² Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/training_analysis/learning_curves/06_{task_name}_learning_curves.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. Feature Importance
    fig, ax = plt.subplots(figsize=(12, 8))
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        ax.barh(range(len(indices)), importances[indices], color='lightgreen', alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f'🌳 Feature Importance - {task_name.title()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/training_analysis/feature_importance/07_{task_name}_feature_importance.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 4. Predictions vs Actual
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use the best model for prediction plot
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_model.fit(x_train, y_train)
    y_pred_best = best_model.predict(x_test)
    
    ax.scatter(y_test, y_pred_best, alpha=0.6, color='blue', s=50)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_title(f'🎯 Predictions vs Actual - {task_name.title()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score to the plot
    r2_best = r2_score(y_test, y_pred_best)
    ax.text(0.05, 0.95, f'R² = {r2_best:.3f}', transform=ax.transAxes, 
             bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/training_analysis/predictions/08_{task_name}_predictions_vs_actual.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 5. Residual Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_test - y_pred_best
    ax.scatter(y_pred_best, residuals, alpha=0.6, color='green', s=50)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    ax.set_title(f'📊 Residual Analysis - {task_name.title()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/training_analysis/residuals/09_{task_name}_residual_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_model_performance_graphs(results_dict, task_name):
    """Create individual model performance graphs"""
    print(f"Creating model performance graphs for {task_name}...")
    
    models = list(results_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # 1. MAE Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    mae_scores = [results_dict[model]['MAE'] for model in models]
    bars1 = ax.bar(range(len(models)), mae_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title(f'📊 Mean Absolute Error - {task_name.title()}\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(' (Enhanced)', '').replace(' (Optimized)', '') for m in models], 
                        rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, mae_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/model_performance/metrics/10_{task_name}_mae_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. R² Score Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    r2_scores = [results_dict[model]['R2'] for model in models]
    bars2 = ax.bar(range(len(models)), r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title(f'🎯 R² Score - {task_name.title()}\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('R²', fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(' (Enhanced)', '').replace(' (Optimized)', '') for m in models], 
                        rotation=45, ha='right')
    
    for i, (bar, value) in enumerate(zip(bars2, r2_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/model_performance/metrics/11_{task_name}_r2_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. Success Rate ±10%
    fig, ax = plt.subplots(figsize=(12, 8))
    success_10 = [results_dict[model]['Success Rate (±10%)'] for model in models]
    bars3 = ax.bar(range(len(models)), success_10, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title(f'✅ Success Rate ±10% - {task_name.title()}\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(' (Enhanced)', '').replace(' (Optimized)', '') for m in models], 
                        rotation=45, ha='right')
    
    for i, (bar, value) in enumerate(zip(bars3, success_10)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/model_performance/success_rates/12_{task_name}_success_rate_10_percent.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 4. Success Rate ±15%
    fig, ax = plt.subplots(figsize=(12, 8))
    success_15 = [results_dict[model]['Success Rate (±15%)'] for model in models]
    bars4 = ax.bar(range(len(models)), success_15, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title(f'✅ Success Rate ±15% - {task_name.title()}\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(' (Enhanced)', '').replace(' (Optimized)', '') for m in models], 
                        rotation=45, ha='right')
    
    for i, (bar, value) in enumerate(zip(bars4, success_15)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/model_performance/success_rates/13_{task_name}_success_rate_15_percent.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 5. Success Rate ±20%
    fig, ax = plt.subplots(figsize=(12, 8))
    success_20 = [results_dict[model]['Success Rate (±20%)'] for model in models]
    bars5 = ax.bar(range(len(models)), success_20, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_title(f'✅ Success Rate ±20% - {task_name.title()}\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace(' (Enhanced)', '').replace(' (Optimized)', '') for m in models], 
                        rotation=45, ha='right')
    
    for i, (bar, value) in enumerate(zip(bars5, success_20)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/model_performance/success_rates/14_{task_name}_success_rate_20_percent.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 6. Radar Chart
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics for radar chart (0-1 scale)
    mae_norm = 1 - (np.array(mae_scores) - np.min(mae_scores)) / (np.max(mae_scores) - np.min(mae_scores))
    r2_norm = (np.array(r2_scores) - np.min(r2_scores)) / (np.max(r2_scores) - np.min(r2_scores))
    success_norm = np.array(success_15) / 100
    
    categories = ['MAE\n(Lower Better)', 'R²\n(Higher Better)', 'Success Rate\n(Higher Better)']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, model in enumerate(models):
        values = [mae_norm[i], r2_norm[i], success_norm[i]]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=model.replace(' (Enhanced)', '').replace(' (Optimized)', ''), 
                color=colors[i], alpha=0.7)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(f'🎯 Model Performance Radar Chart - {task_name.title()}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/model_performance/radar_charts/15_{task_name}_performance_radar_chart.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 7. Best Model Highlight
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Find best model based on R² score
    best_model_idx = np.argmax(r2_scores)
    best_model = models[best_model_idx]
    
    # Create a summary box
    ax.text(0.5, 0.8, f'🏆 BEST MODEL', ha='center', va='center', 
             fontsize=18, fontweight='bold', color='gold')
    ax.text(0.5, 0.6, best_model, ha='center', va='center', 
             fontsize=14, fontweight='bold', wrap=True)
    ax.text(0.5, 0.4, f'R²: {r2_scores[best_model_idx]:.3f}', ha='center', va='center', 
             fontsize=12, fontweight='bold')
    ax.text(0.5, 0.2, f'Success Rate: {success_15[best_model_idx]:.1f}%', ha='center', va='center', 
             fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='gold', linewidth=3))
    ax.set_title(f'Best Model - {task_name.title()}', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'images/individual_graphs/model_performance/best_models/16_{task_name}_best_model_highlight.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Main function to create individual graphs"""
    print("Creating Individual Graphs from EV Dataset Analysis")
    print("=" * 60)
    
    # Create folders
    create_individual_graph_folders()
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Create dataset overview graphs
    create_dataset_overview_graphs(df)
    # Create additional graphs to reach 12 individual panels like the dashboard
    create_additional_exploration_graphs(df)
    
    # Prepare features and targets
    feature_cols = [
        TRIP_DISTANCE_COL, AVG_SPEED_COL, 'city', 'motor_way', 
        'country_roads', 'A/C', 'park_heating', 'tire_type', 
        'driving_style', POWER_COL, 'fuel_type',
        'energy_efficiency', 'power_density', 'road_diversity', 'climate_impact',
        'speed_distance_interaction', 'power_speed_interaction', 'distance_efficiency_interaction',
        'speed_squared', 'distance_squared', 'power_squared',
        'speed_category', 'distance_category'
    ]
    
    target_cols = {
        'consumption': CONSUMPTION_COL,
        'quantity': QUANTITY_COL, 
        'ecr_deviation': ECR_DEVIATION_COL
    }
    
    X = df[feature_cols].copy()
    
    categorical_features = ['tire_type', 'driving_style', 'fuel_type', 'speed_category', 'distance_category']
    numerical_features = [col for col in feature_cols if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    x_processed = preprocessor.fit_transform(X)
    
    # Get models
    models = {
        'Ridge (Optimized)': Ridge(alpha=0.1),
        'Lasso (Optimized)': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'Random Forest (Enhanced)': RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42
        ),
        'Gradient Boosting (Enhanced)': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            min_samples_split=10, min_samples_leaf=4, random_state=42
        ),
        'MLP (Enhanced)': MLPRegressor(
            hidden_layer_sizes=(200, 100, 50), activation='relu', solver='adam',
            alpha=0.001, learning_rate='adaptive', max_iter=1000, random_state=42
        )
    }
    
    # Create ensemble models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ('ridge', Ridge(alpha=0.1))
    ]
    
    ensemble_models = {
        'Voting Ensemble': VotingRegressor(base_models),
        'Stacking Ensemble': StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=0.1),
            cv=5
        )
    }
    
    all_models = {**models, **ensemble_models}
    
    # Train and evaluate for each task
    all_results = {}
    
    for task_name, target_col in target_cols.items():
        y = df[target_col].copy()
        
        # Create training analysis graphs
        create_training_analysis_graphs(x_processed, y, task_name, all_models)
        
        # Train and evaluate models
        valid_idx = ~y.isna()
        x_clean = x_processed[valid_idx]
        y_clean = y[valid_idx]
        
        x_train, x_test, y_train, y_test = train_test_split(
            x_clean, y_clean, test_size=0.2, random_state=42
        )
        
        results = {}
        best_model_name = None
        best_r2 = -np.inf
        best_y_test = None
        best_y_pred = None
        for name, model in all_models.items():
            try:
                cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                success_rates = {}
                for threshold in [0.05, 0.10, 0.15, 0.20]:
                    success_mask = (np.abs(y_test - y_pred) / np.abs(y_test)) <= threshold
                    success_rate = success_mask.sum() / len(y_test) * 100
                    success_rates[f'Success Rate (±{int(threshold*100)}%)'] = success_rate
                
                results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'CV_R2_Mean': cv_scores.mean(),
                    'CV_R2_Std': cv_scores.std(),
                    **success_rates
                }
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = name
                    best_y_test = y_test
                    best_y_pred = y_pred
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        all_results[task_name] = results
        
        # Create model performance graphs
        if results:
            create_model_performance_graphs(results, task_name)
            # Save confusion matrix for the best model
            if best_y_test is not None and best_y_pred is not None:
                save_confusion_matrix_binned(best_y_test, best_y_pred, task_name)
    
    print("Individual graphs created successfully!")
    print("All graphs saved in images/individual_graphs/ folder")
    print("Each graph is saved as a separate high-quality PNG file")

if __name__ == "__main__":
    main()
