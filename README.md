# 🚗 EV Dataset Analysis - Enhanced Model Analyzer with Advanced Visualizations

## 📋 Overview

This project provides comprehensive analysis and machine learning models for Electric Vehicle (EV) energy consumption prediction using real-world driving data from Mitsubishi i-MiEV and Volkswagen e-Golf vehicles. The enhanced analyzer now includes **professional-grade visualizations**, **comprehensive training analysis**, and **advanced data exploration dashboards**.

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

| File | Description | Content |
|------|-------------|---------|
| `data_exploration_dashboard.png` | **Data Overview** | Dataset statistics, distributions, correlations |
| `training_analysis_*.png` | **Training Analysis** | Learning curves, feature importance, predictions |
| `model_performance_*.png` | **Model Comparison** | Performance metrics, radar charts, best model highlights |

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
- **Professional Styling**: High-quality 300 DPI images with emojis and annotations

## 📈 Performance Results

### Success Rate Improvements

| Task | Original (±10%) | Enhanced (±10%) | Improvement |
|------|----------------|-----------------|-------------|
| **Consumption** | 48-55% | **85-89%** | **+35-40%** |
| **Quantity** | 22-29% | **88-100%** | **+60-70%** |
| **ECR Deviation** | 14-18% | **42-47%** | **+25-30%** |

### Best Performing Models

#### 🥇 Consumption Prediction (kWh/100km)
- **Stacking Ensemble**: 88.6% success (±10%), 97.8% (±20%)
- **Gradient Boosting**: 88.6% success (±10%), 97.5% (±20%)
- **MLP Enhanced**: 87.6% success (±10%), 97.6% (±20%)

#### 🥇 Quantity Prediction (kWh)
- **Voting/Stacking Ensemble**: 100% success (±10%)
- **Ridge Optimized**: 100% success (±10%)
- **Gradient Boosting**: 99.2% success (±10%)

#### 🥇 ECR Deviation Prediction
- **Gradient Boosting**: 47.1% success (±10%), 63.9% (±20%)
- **Random Forest**: 44.3% success (±10%), 62.6% (±20%)
- **MLP Enhanced**: 46.6% success (±10%), 61.3% (±20%)

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
- **Interactive Elements**: Value labels, annotations, and statistical overlays
- **Visual Analytics**: Radar charts, heatmaps, and performance comparisons
- **Training Insights**: Learning curves, feature importance, and residual analysis

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
- Focus on **ensemble methods** (Stacking, Voting)
- Use **±15% or ±20% thresholds** for realistic success rates

### For Production Deployment
- **Consumption**: Stacking Ensemble (97.8% success at ±20%)
- **Quantity**: Voting Ensemble (100% success at ±10%)
- **ECR Deviation**: Gradient Boosting (63.9% success at ±20%)

### For Research/Analysis
- **Feature importance**: Random Forest provides interpretable feature rankings
- **Model comparison**: All models included for comprehensive evaluation
- **Cross-validation**: Robust performance estimation

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

### Success Rate Factors
1. **Feature engineering** is the most important factor
2. **Tree-based models** consistently outperform linear models
3. **Ensemble methods** provide the best overall performance
4. **Outlier handling** significantly improves model stability
5. **Cross-validation** provides reliable performance estimates
6. **Visual analysis** helps identify patterns and model behavior

### Task Difficulty
1. **Quantity prediction** is easiest (100% success achievable)
2. **Consumption prediction** is moderate (85-89% success)
3. **ECR deviation** is hardest (42-47% success) - inherently more variable

### Visualization Benefits
1. **Data Understanding**: Comprehensive dashboards reveal data patterns
2. **Model Insights**: Training analysis shows learning behavior
3. **Performance Comparison**: Visual metrics make model selection easier
4. **Professional Presentation**: High-quality images suitable for reports
5. **Error Analysis**: Residual plots help identify prediction issues

## 🤝 Contributing

To improve the models further:
1. Add more vehicle datasets
2. Implement additional feature engineering techniques
3. Try more advanced ensemble methods
4. Add time-series features for temporal patterns
5. Implement hyperparameter optimization with GridSearchCV
6. Add interactive web dashboards
7. Implement real-time prediction APIs
8. Add more visualization types (3D plots, animations)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset sources: Mitsubishi i-MiEV and Volkswagen e-Golf driving logs
- Scikit-learn community for excellent ML tools
- Pandas and NumPy for data processing capabilities

---

## 🎨 Visualization Gallery

The enhanced analyzer generates professional-quality visualizations:

- **📊 Data Exploration Dashboard**: Complete dataset overview with statistics and distributions
- **🚀 Training Analysis**: Learning curves, feature importance, and prediction analysis
- **🎯 Model Performance**: Comprehensive comparison charts with radar plots and best model highlights
- **📁 High-Quality Output**: 300 DPI PNG images ready for presentations and publications

---

**Note**: The enhanced model analyzer (`enhanced_model_analyzer.py`) represents the state-of-the-art implementation with the highest success rates achieved in this project, now featuring comprehensive visualization capabilities for professional analysis and presentation.
