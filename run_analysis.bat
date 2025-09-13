@echo off
echo ================================================================
echo  🚗 EV Dataset Analysis - Enhanced Model Analyzer with Visualizations 🚗
echo ================================================================
echo.
echo 🎯 This enhanced analyzer provides:
echo      . Advanced Feature Engineering
echo      . Comprehensive Data Exploration Dashboard
echo      . Training Analysis with Learning Curves
echo      . Model Performance Comparison Charts
echo      . High-Quality Visualizations (300 DPI)
echo      . Multiple Success Rate Thresholds
echo     Ensemble Methods (Voting and Stacking)
echo.
echo 📊 Generated Visualizations:
echo    • data_exploration_dashboard.png
echo    • training_analysis_*.png (for each target variable)
echo    • model_performance_*.png (for each target variable)
echo.
echo 🚀 Starting comprehensive analysis...
echo ================================================================
echo.
python enhanced_model_analyzer.py
echo.
echo ================================================================
echo 🎉 ANALYSIS COMPLETE! 🎉
echo ================================================================
echo.
echo 📁 Check the current directory for generated PNG files:
echo    • High-quality plots saved at 300 DPI resolution
echo    • Professional styling with emojis and annotations
echo    • Comprehensive analysis dashboards
echo.
echo 💡 Key Features Analyzed:
echo    • Consumption prediction (kWh/100km)
echo    • Energy quantity prediction (kWh)
echo    • ECR deviation analysis
echo.
echo Press any key to exit...
pause
