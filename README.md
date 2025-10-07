# 🚗 EV Dataset Analysis - Enhanced Model Analyzer with Advanced Visualizations

## 📋 Overview

This project provides comprehensive analysis and machine learning models for Electric Vehicle (EV) energy consumption prediction using real-world driving data from Mitsubishi i-MiEV and Volkswagen e-Golf vehicles. The enhanced analyzer now includes **professional-grade visualizations**, **comprehensive training analysis**, **advanced data exploration dashboards**, **individual graph generation**, and **confusion matrix analysis**.

## 🎯 Project Goals

- **Predict Energy Consumption** (kWh/100km) - Energy efficiency per 100km
- **Predict Trip Energy** (quantity kWh) - Total energy consumption per trip
- **Predict ECR Deviation** - Energy consumption rate deviation from expected values

## 📊 Dataset Information

- **Combined Dataset**: 4,591 samples from 2 vehicle models
- **Mitsubishi i-MiEV**: 1,285 records (20 columns)
- **Volkswagen e-Golf**: 3,345 records (18 columns)
- **Features**: Trip distance, speed, road types, climate control, vehicle specs, etc.

## 🚀 Project Structure

### Core Files

| File | Description | Purpose |
|------|-------------|---------|
| `enhanced_model_analyzer.py` | **Main analyzer** | Advanced ML models with comprehensive visualizations |
| `README.md` | **Documentation** | Complete project guide and results |
| `run_analysis.bat` | **Run script** | Easy execution on Windows with enhanced features |

### Generated Visualizations

All visualizations are now organized in the `images/` folder with comprehensive documentation. In addition to the dashboard-style images, the project now generates **INDIVIDUAL graphs** for every panel (data exploration, training analysis, and model performance) under `images/individual_graphs/`. The enhanced analyzer also creates an `individual/` folder with organized individual graphs and **confusion matrices** for regression analysis.

| Folder | Description | Files | Content |
|--------|-------------|-------|---------|
| `images/data_exploration/` | **Dashboards** | 1 file | Dataset overview dashboard |
| `images/training_analysis/` | **Dashboards** | 3 files | Per-task training analysis dashboards |
| `images/model_performance/` | **Dashboards** | 3 files | Per-task model comparison dashboards |
| `images/individual_graphs/` | **Individual panels** | many | One PNG per panel, organized by category & task |
| `individual/` | **Enhanced individual graphs** | many | Organized individual graphs + confusion matrices |

#### 📁 Detailed Image Organization

**Data Exploration** (`images/data_exploration/`)
- `01_dataset_overview_dashboard.png` - Complete dataset analysis and statistical overview

**Individual Data Exploration Panels** (`images/individual_graphs/data_exploration/`)
- `dataset_overview/01_dataset_statistics.png`
- `target_distributions/02_consumptionkWh_100km_distribution.png`
- `target_distributions/02_quantitykWh_distribution.png`
- `target_distributions/02_ecr_deviation_distribution.png`
- `correlations/03_feature_correlation_heatmap.png`
- `missing_values/04_missing_values_analysis.png`
- `feature_distributions/05_power_distribution.png`
- `feature_distributions/06_trip_distance_distribution.png`
- `categorical_distributions/09_city_distribution.png`
- `categorical_distributions/10_manufacturer_distribution.png`
- `categorical_distributions/11_model_distribution.png`
- `categorical_distributions/12_version_distribution.png`

**Training Analysis** (`images/training_analysis/`)
- `02_consumption_training_analysis.png` - Energy consumption training insights
- `03_ecr_deviation_training_analysis.png` - ECR deviation training insights  
- `04_quantity_training_analysis.png` - Trip energy quantity training insights

**Individual Training Analysis Panels** (`images/individual_graphs/training_analysis/`)
- For each task `{consumption|quantity|ecr_deviation}`:
  - `data_distribution/05_{task}_target_distribution.png`
  - `learning_curves/06_{task}_learning_curves.png`
  - `feature_importance/07_{task}_feature_importance.png`
  - `predictions/08_{task}_predictions_vs_actual.png`
  - `residuals/09_{task}_residual_analysis.png`

**Model Performance** (`images/model_performance/`)
- `05_consumption_model_comparison.png` - Consumption model performance comparison
- `06_ecr_deviation_model_comparison.png` - ECR deviation model performance comparison
- `07_quantity_model_comparison.png` - Quantity model performance comparison

**Individual Model Performance Panels** (`images/individual_graphs/model_performance/`)
- For each task `{consumption|quantity|ecr_deviation}`:
  - `metrics/10_{task}_mae_comparison.png`
  - `metrics/11_{task}_r2_comparison.png`
  - `success_rates/12_{task}_success_rate_10_percent.png`
  - `success_rates/13_{task}_success_rate_15_percent.png`
  - `success_rates/14_{task}_success_rate_20_percent.png`
  - `radar_charts/15_{task}_performance_radar_chart.png`
  - `best_models/16_{task}_best_model_highlight.png`

**Enhanced Individual Graphs** (`individual/` folder)
- **Data Exploration**: `individual/data_exploration/` - Dataset overview, target distributions, correlations, missing values
- **Training Analysis**: `individual/training_analysis/` - Data distribution, learning curves, feature importance, predictions, residuals
- **Model Performance**: `individual/model_performance/` - MAE/R² comparisons, success rates, radar charts, best model highlights
- **Confusion Matrices**: `individual/model_performance/confusion_matrices/` - Binned confusion matrices for best models per task

Each folder contains detailed README.md files explaining the purpose, content, and insights of every visualization.

### Data Files

| File | Description | Records | Columns |
|------|-------------|---------|---------|
| `mitsubishi_imiev.csv` | Mitsubishi i-MiEV driving data | 1,285 | 20 |
| `volkswagen_e_golf.csv` | Volkswagen e-Golf driving data | 3,345 | 18 |

## 🔧 Installation Requirements

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Optional Packages (for enhanced performance)
```bash
pip install xgboost lightgbm tensorflow
```

## 🎨 New Visualization Features

### 📊 Data Exploration Dashboard
- **Dataset Overview**: Statistics, memory usage, data types
- **Target Variable Distributions**: Histograms with statistical overlays
- **Feature Correlation Matrix**: Color-coded correlation heatmap
- **Missing Values Analysis**: Visual missing data patterns
- **Feature Distributions**: Numeric and categorical feature analysis

### 🚀 Training Analysis Dashboard
- **Data Distribution**: Target variable histograms with mean/std lines
- **Feature Correlation Heatmap**: Inter-feature relationships
- **Learning Curves**: Training vs validation performance over data size
- **Model Performance Comparison**: Side-by-side model evaluation
- **Prediction vs Actual Scatter Plots**: Perfect prediction line analysis
- **Residual Analysis**: Error pattern detection
- **Feature Importance**: Random Forest feature rankings
- **Error Distribution**: Residual histograms
- **Training Progress Simulation**: Overfitting detection

### 🎯 Enhanced Model Performance Plots
- **Comprehensive Metrics**: MAE, RMSE, R², Cross-validation scores
- **Success Rate Analysis**: Multiple threshold comparisons (±10%, ±15%, ±20%)
- **Radar Chart**: Multi-dimensional model comparison
- **Best Model Highlight**: Performance summary with visual emphasis
- **Confusion Matrices**: Binned confusion matrices for regression analysis
- **Professional Styling**: High-quality 300 DPI images with emojis and annotations

## 📈 Performance Results

### 🎯 Comprehensive Model Performance Summary

| Task | Model | MAE | RMSE | R² | CV R² | Success ±10% | Success ±15% | Success ±20% |
|------|-------|-----|------|----|-------|--------------|--------------|--------------|
| **consumption** | Ridge (Optimized) | 2.116 | 2.758 | 0.274 | 0.261 | 44.3% | 59.2% | 69.4% |
| **consumption** | Lasso (Optimized) | 2.114 | 2.772 | 0.267 | 0.255 | 45.8% | 59.1% | 68.8% |
| **consumption** | ElasticNet | 2.113 | 2.769 | 0.268 | 0.257 | 45.5% | 59.1% | 69.0% |
| **consumption** | Random Forest (Enhanced) | 0.630 | 1.001 | 0.904 | 0.900 | **85.6%** | **93.5%** | **97.0%** |
| **consumption** | Gradient Boosting (Enhanced) | 0.559 | 1.002 | 0.904 | 0.906 | **88.6%** | **94.5%** | **97.5%** |
| **consumption** | MLP (Enhanced) | 0.666 | 0.988 | 0.907 | 0.864 | **87.6%** | **94.6%** | **97.6%** |
| **consumption** | Voting Ensemble | 0.992 | 1.288 | 0.842 | 0.837 | 71.4% | 85.9% | 91.3% |
| **consumption** | Stacking Ensemble | 0.549 | 0.967 | 0.911 | 0.912 | **88.6%** | **94.1%** | **97.8%** |
| **quantity** | Ridge (Optimized) | 0.002 | 0.022 | 1.000 | 1.000 | **100.0%** | **100.0%** | **100.0%** |
| **quantity** | Lasso (Optimized) | 0.031 | 0.399 | 0.992 | 0.994 | 97.9% | 98.8% | 99.1% |
| **quantity** | ElasticNet | 0.064 | 0.377 | 0.993 | 0.994 | 94.9% | 96.8% | 98.5% |
| **quantity** | Random Forest (Enhanced) | 0.216 | 0.486 | 0.988 | 0.985 | 88.0% | 92.2% | 94.5% |
| **quantity** | Gradient Boosting (Enhanced) | 0.011 | 0.065 | 1.000 | 0.998 | **99.2%** | **99.6%** | **99.8%** |
| **quantity** | MLP (Enhanced) | 0.102 | 0.290 | 0.996 | 0.996 | 89.1% | 91.5% | 92.6% |
| **quantity** | Voting Ensemble | 0.008 | 0.018 | 1.000 | 1.000 | **100.0%** | **100.0%** | **100.0%** |
| **quantity** | Stacking Ensemble | 0.004 | 0.021 | 1.000 | 1.000 | **100.0%** | **100.0%** | **100.0%** |
| **ecr_deviation** | Ridge (Optimized) | 2.201 | 3.361 | 0.211 | 0.225 | 12.2% | 17.4% | 21.9% |
| **ecr_deviation** | Lasso (Optimized) | 2.187 | 3.369 | 0.207 | 0.224 | 12.3% | 18.4% | 22.9% |
| **ecr_deviation** | ElasticNet | 2.191 | 3.369 | 0.207 | 0.223 | 12.2% | 18.2% | 22.2% |
| **ecr_deviation** | Random Forest (Enhanced) | 0.729 | 2.190 | 0.665 | 0.795 | **44.3%** | **54.8%** | **62.6%** |
| **ecr_deviation** | Gradient Boosting (Enhanced) | 0.689 | 2.250 | 0.646 | 0.779 | **47.1%** | **55.5%** | **63.9%** |
| **ecr_deviation** | MLP (Enhanced) | 0.708 | 2.293 | 0.632 | 0.795 | **46.6%** | **55.8%** | **61.3%** |
| **ecr_deviation** | Voting Ensemble | 1.080 | 2.327 | 0.621 | 0.736 | 23.5% | 32.6% | 43.1% |
| **ecr_deviation** | Stacking Ensemble | 0.714 | 2.200 | 0.662 | 0.781 | **42.4%** | **54.0%** | **60.9%** |

### 🏆 Best Performing Models by Task

#### 🥇 Consumption Prediction (kWh/100km)
- **🥇 Stacking Ensemble**: 88.6% success (±10%), 97.8% (±20%) - **Best Overall**
- **🥈 Gradient Boosting**: 88.6% success (±10%), 97.5% (±20%) - **Excellent Performance**
- **🥉 MLP Enhanced**: 87.6% success (±10%), 97.6% (±20%) - **Strong Neural Network**

#### 🥇 Quantity Prediction (kWh)
- **🥇 Ridge/Voting/Stacking Ensemble**: 100% success (±10%) - **Perfect Performance**
- **🥈 Gradient Boosting**: 99.2% success (±10%), 99.8% (±20%) - **Near Perfect**
- **🥉 Lasso Optimized**: 97.9% success (±10%), 99.1% (±20%) - **Very Strong**

#### 🥇 ECR Deviation Prediction
- **🥇 Gradient Boosting**: 47.1% success (±10%), 63.9% (±20%) - **Best Performance**
- **🥈 MLP Enhanced**: 46.6% success (±10%), 61.3% (±20%) - **Strong Neural Network**
- **🥉 Random Forest**: 44.3% success (±10%), 62.6% (±20%) - **Solid Ensemble**

## 🛠️ Key Improvements Implemented

### 1. Advanced Feature Engineering
- **Energy efficiency ratio**: `quantity(kWh) / trip_distance(km)`
- **Speed categories**: city, urban, highway, motorway
- **Distance categories**: short, medium, long, very_long
- **Power density**: power-to-weight ratio
- **Road diversity**: combination of road types
- **Climate impact**: A/C + park heating effects
- **Interaction features**: speed×distance, power×speed
- **Polynomial features**: squared terms for key variables

### 2. Data Preprocessing Enhancements
- **European number format handling**: Comma to dot conversion
- **Outlier handling**: IQR method with capping instead of removal
- **RobustScaler**: Better outlier handling than StandardScaler
- **Missing value imputation**: Median for numeric, 'Unknown' for categorical
- **Categorical encoding**: One-hot encoding with proper handling

### 3. Model Optimization
- **Hyperparameter tuning**: Optimized parameters for each model
- **Cross-validation**: 5-fold CV for robust evaluation
- **Ensemble methods**: Voting and Stacking regressors
- **Multiple success thresholds**: ±10%, ±15%, ±20%

### 4. Model Architecture Improvements
- **Random Forest**: 300 estimators, max_depth=20, optimized splits
- **Gradient Boosting**: 300 estimators, learning_rate=0.05, max_depth=8
- **MLP**: 3 hidden layers (200,100,50), adaptive learning
- **Ensemble**: Voting and Stacking with meta-learners

### 5. 🎨 Advanced Visualization System
- **Professional Styling**: Consistent color schemes, fonts, and layouts
- **High-Quality Output**: 300 DPI PNG images for publication quality
- **Comprehensive Dashboards**: Multi-panel analysis views
- **Individual Graph Generation**: Separate PNG files for each visualization panel
- **Confusion Matrix Analysis**: Binned confusion matrices for regression tasks
- **Interactive Elements**: Value labels, annotations, and statistical overlays
- **Visual Analytics**: Radar charts, heatmaps, and performance comparisons
- **Training Insights**: Learning curves, feature importance, and residual analysis
- **Organized Structure**: Both dashboard and individual graph outputs

## 🚀 Quick Start

### Run Enhanced Analysis (Recommended)
```bash
python enhanced_model_analyzer.py
```

### Run with Enhanced Batch Script (Windows)
```bash
run_analysis.bat
```

### What You'll Get
- **📊 Data Exploration Dashboard**: Complete dataset analysis
- **🚀 Training Analysis**: Learning curves and model insights for each target
- **🎯 Performance Comparison**: Comprehensive model evaluation charts
- **📁 Individual Graphs**: Separate PNG files for each visualization panel
- **🔍 Confusion Matrices**: Binned confusion matrices for regression analysis
- **📁 High-Quality Images**: Professional 300 DPI PNG files ready for presentations

## 📊 Model Comparison

### Available Models

#### Baseline Models
- **Linear Regression**: Fast, interpretable baseline
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization with feature selection
- **ElasticNet**: Combined L1+L2 regularization

#### Tree-Based Models (Best Performance)
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential boosting
- **XGBoost**: Optimized gradient boosting (optional)
- **LightGBM**: Fast gradient boosting (optional)

#### Neural Networks
- **MLP Regressor**: Multi-layer perceptron
- **Keras Sequential**: Deep learning model (optional)

#### Ensemble Methods
- **Voting Ensemble**: Combines multiple models
- **Stacking Ensemble**: Uses meta-learner for final prediction

## 🔍 Feature Analysis

### Original Features
- `trip_distance(km)`: Distance of the trip
- `avg_speed(km/h)`: Average speed during trip
- `city`, `motor_way`, `country_roads`: Road type indicators
- `A/C`, `park_heating`: Climate control usage
- `tire_type`, `driving_style`: Categorical variables
- `power(kW)`: Vehicle rated power

### Engineered Features
- `energy_efficiency`: Energy per distance ratio
- `speed_category`: Categorized speed ranges
- `distance_category`: Categorized distance ranges
- `power_density`: Power-to-weight ratio
- `road_diversity`: Combined road type indicator
- `climate_impact`: Combined climate control usage
- `speed_distance_interaction`: Speed × distance
- `power_speed_interaction`: Power × speed
- `distance_efficiency_interaction`: Distance × efficiency
- `speed_squared`, `distance_squared`, `power_squared`: Polynomial terms

## 📈 Evaluation Metrics

### Primary Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference
- **RMSE (Root Mean Square Error)**: Square root of mean squared error
- **R² Score**: Coefficient of determination
- **Success Rate**: Percentage of predictions within ±10%, ±15%, ±20% of actual values

### Cross-Validation
- **5-fold CV**: Robust evaluation across different data splits
- **CV R² Mean**: Average R² across folds
- **CV R² Std**: Standard deviation of R² scores

## 🎯 Usage Recommendations

### For Maximum Accuracy
- Use **enhanced_model_analyzer.py** for best results
- Focus on **ensemble methods** (Stacking, Voting) - proven top performers
- Use **±15% or ±20% thresholds** for realistic success rates
- **Hyperparameter optimization** essential for peak performance

### For Production Deployment
- **Consumption**: **Stacking Ensemble** (88.6% success at ±10%, 97.8% at ±20%) - **Recommended**
- **Quantity**: **Ridge/Voting/Stacking Ensemble** (100% success at ±10%) - **Perfect Performance**
- **ECR Deviation**: **Gradient Boosting** (47.1% success at ±10%, 63.9% at ±20%) - **Best Available**

### For Research/Analysis
- **Feature importance**: Random Forest provides interpretable feature rankings
- **Model comparison**: All models included for comprehensive evaluation
- **Cross-validation**: Robust performance estimation with CV R² scores
- **Visual analysis**: Comprehensive dashboards and individual graphs for insights
- **Confusion matrices**: Binned analysis for regression model evaluation

### 🚀 Latest Analysis Status
✅ **Analysis Complete** - All models trained and evaluated
✅ **Visualizations Generated** - Professional 300 DPI images created
✅ **Individual Graphs** - Separate PNG files for each visualization panel
✅ **Confusion Matrices** - Binned confusion matrices for regression analysis
✅ **Performance Metrics** - Comprehensive evaluation across all success thresholds

## 🔧 Technical Details

### Data Processing Pipeline
1. **Load datasets** with encoding detection
2. **Align columns** between datasets
3. **Feature engineering** with advanced transformations
4. **Preprocessing** with scaling and encoding
5. **Train/test split** (80/20)
6. **Model training** with cross-validation
7. **Evaluation** with multiple metrics
8. **Visualization** with performance plots

### Model Selection Strategy
1. **Start with tree-based models** (Random Forest, Gradient Boosting)
2. **Try ensemble methods** (Voting, Stacking)
3. **Use linear models** for baseline comparison
4. **Apply neural networks** for complex patterns
5. **Cross-validate** all models for robust evaluation

## 📝 Key Findings

### 🎯 Latest Analysis Results (2024)

#### Success Rate Achievements
- **Consumption Prediction**: Achieved up to **88.6% success** at ±10% threshold, **97.8%** at ±20%
- **Quantity Prediction**: Achieved **100% success** at ±10% threshold - **Perfect Performance**
- **ECR Deviation Prediction**: Achieved up to **47.1% success** at ±10% threshold, **63.9%** at ±20%

#### Model Performance Insights
1. **Stacking Ensemble** emerges as the top performer for consumption prediction
2. **Ridge Regression** achieves perfect performance for quantity prediction (R² = 1.000)
3. **Gradient Boosting** consistently delivers strong performance across all tasks
4. **Enhanced models** significantly outperform baseline linear models
5. **Cross-validation** confirms robust performance with CV R² scores close to test R²

#### Task Difficulty Analysis
1. **Quantity prediction** is easiest (100% success achievable) - highly predictable
2. **Consumption prediction** is moderate (85-89% success) - good predictability with advanced models
3. **ECR deviation** is hardest (42-47% success) - inherently more variable and challenging

### Success Rate Factors
1. **Feature engineering** is the most important factor - advanced interactions crucial
2. **Tree-based models** consistently outperform linear models by 40-60% success rate
3. **Ensemble methods** provide the best overall performance - stacking particularly effective
4. **Outlier handling** significantly improves model stability - RobustScaler essential
5. **Cross-validation** provides reliable performance estimates - prevents overfitting
6. **Visual analysis** helps identify patterns and model behavior - comprehensive dashboards
7. **Hyperparameter optimization** crucial for achieving peak performance
8. **Multiple success thresholds** provide realistic performance expectations

### Visualization Benefits
1. **Data Understanding**: Comprehensive dashboards reveal data patterns and quality
2. **Model Insights**: Training analysis shows learning behavior and overfitting detection
3. **Performance Comparison**: Visual metrics make model selection easier and more intuitive
4. **Professional Presentation**: High-quality 300 DPI images suitable for reports and publications
5. **Error Analysis**: Residual plots help identify prediction issues and model limitations
6. **Individual Analysis**: Separate graphs for detailed examination of each component
7. **Confusion Matrix Analysis**: Binned confusion matrices for regression model evaluation
8. **Organized Output**: Both comprehensive dashboards and individual graphs for different use cases
9. **Training Insights**: Learning curves reveal model behavior and data sufficiency
10. **Feature Importance**: Visual rankings help understand model decision-making

### 🔬 Technical Insights
1. **R² Scores**: Quantity prediction achieves perfect R² = 1.000, indicating excellent linear relationships
2. **MAE Improvements**: Enhanced models reduce MAE by 60-80% compared to baseline models
3. **Cross-Validation Stability**: CV R² scores closely match test R², indicating robust performance
4. **Ensemble Effectiveness**: Stacking ensemble particularly effective for complex consumption prediction
5. **Neural Network Performance**: MLP models competitive with tree-based methods
6. **Success Rate Scaling**: Performance improves dramatically with relaxed thresholds (±15%, ±20%)

## 🎉 Analysis Completion Summary

### 🔧 Key Improvements Implemented
✅ **Advanced feature engineering** with interactions and polynomial features
✅ **Outlier handling** with RobustScaler for better stability
✅ **Hyperparameter optimization** for peak model performance
✅ **Ensemble methods** (Voting & Stacking) for superior accuracy
✅ **Cross-validation** for robust evaluation and overfitting prevention
✅ **Multiple success rate thresholds** (±10%, ±15%, ±20%) for realistic assessment
✅ **Comprehensive data visualization** with professional 300 DPI images
✅ **Training analysis dashboards** for model behavior insights
✅ **Individual graph generation** for detailed component analysis
✅ **Confusion matrix analysis** for regression model evaluation

### 💡 Success Rate Improvement Strategies
1. **Use ±15% or ±20% thresholds** for more realistic success rates
2. **Focus on ensemble methods** for better predictions - Stacking particularly effective
3. **Consider task difficulty** - some predictions are inherently harder (ECR deviation)
4. **Feature engineering significantly improves** model performance - crucial factor
5. **Visual analysis helps identify** patterns and outliers in data
6. **Training analysis reveals** model behavior and overfitting issues
7. **Hyperparameter optimization** essential for achieving peak performance
8. **Cross-validation provides** reliable performance estimates

### 📊 Generated Outputs
- **📁 All plots saved** as high-quality PNG images (300 DPI)
- **📊 Comprehensive dashboards** in current directory
- **📁 Individual graphs** organized in 'individual/' folder structure
- **🔍 Confusion matrices** for regression analysis with binned targets
- **📈 Performance comparisons** across all models and tasks

## 🤝 Contributing

To improve the models further:
1. Add more vehicle datasets for broader generalization
2. Implement additional feature engineering techniques
3. Try more advanced ensemble methods (AdaBoost, Extra Trees)
4. Add time-series features for temporal patterns
5. Implement hyperparameter optimization with GridSearchCV
6. Add interactive web dashboards for real-time analysis
7. Implement real-time prediction APIs
8. Add more visualization types (3D plots, animations)
9. Implement deep learning models (CNN, LSTM) for complex patterns
10. Add model interpretability tools (SHAP, LIME)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset sources: Mitsubishi i-MiEV and Volkswagen e-Golf driving logs
- Scikit-learn community for excellent ML tools
- Pandas and NumPy for data processing capabilities

---

## 🎨 Visualization Gallery

The enhanced analyzer generates professional-quality visualizations, now organized in the `images/` folder:

### 📊 Data Exploration (`images/data_exploration/`)
- **Complete Dataset Overview**: Statistics, distributions, correlations, and data quality analysis
- **Feature Analysis**: Numeric and categorical feature distributions with missing value analysis
- **Correlation Matrix**: Color-coded heatmap showing inter-feature relationships

### 🚀 Training Analysis (`images/training_analysis/`)
- **Learning Curves**: Training vs validation performance over data size
- **Feature Importance**: Random Forest feature rankings for each prediction task
- **Prediction Analysis**: Scatter plots showing prediction vs actual values
- **Residual Analysis**: Error patterns and distribution analysis
- **Training Progress**: Epoch-by-epoch training behavior simulation

### 🎯 Model Performance (`images/model_performance/`)
- **Comprehensive Metrics**: MAE, RMSE, R², and cross-validation scores
- **Success Rate Analysis**: Multiple threshold comparisons (±10%, ±15%, ±20%)
- **Radar Charts**: Multi-dimensional model comparison
- **Best Model Highlights**: Performance summaries with visual emphasis
- **Confusion Matrices**: Binned confusion matrices for regression analysis
- **Professional Styling**: High-quality 300 DPI images with annotations

### 📁 Organization Benefits
- **Clear Structure**: Logical categorization by analysis type
- **Detailed Documentation**: README files explaining each visualization
- **Easy Navigation**: Sequential numbering and descriptive names
- **Publication Ready**: Professional quality suitable for reports and presentations
- **Dual Output System**: Both comprehensive dashboards and individual graphs
- **Confusion Matrix Analysis**: Specialized regression analysis with binned targets

---

**Note**: The enhanced model analyzer (`enhanced_model_analyzer.py`) represents the state-of-the-art implementation with the highest success rates achieved in this project, now featuring comprehensive visualization capabilities, individual graph generation, and confusion matrix analysis for professional analysis and presentation.
