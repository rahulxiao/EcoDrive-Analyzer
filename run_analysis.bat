@echo off
echo ================================================================
echo  🚗 EV Dataset Analysis - Individual Graphs Generator 🚗
echo ================================================================
echo.
echo 🎯 This individual graphs generator provides:
echo      . Advanced Feature Engineering
echo      . Comprehensive Data Exploration Dashboard
echo      . Training Analysis with Learning Curves
echo      . Model Performance Comparison Charts
echo      . High-Quality Visualizations (300 DPI)
echo      . Multiple Success Rate Thresholds
echo      . Ensemble Methods (Voting and Stacking)
echo      . Organized Individual Graph Structure
echo.
echo 📊 Generated Visualizations Structure:
echo    • images/data_exploration/ - Dataset overview dashboard
echo    • images/individual_graphs/data_exploration/ - Individual exploration graphs
echo    • images/individual_graphs/training_analysis/ - Training analysis graphs
echo    • images/individual_graphs/model_performance/ - Model comparison graphs
echo    • images/training_analysis/ - Combined training analysis
echo    • images/model_performance/ - Combined model performance
echo.
echo 🚀 Starting comprehensive analysis...
echo ================================================================
echo.
python create_individual_graphs.py
echo.
echo ================================================================
echo 🎉 ANALYSIS COMPLETE! 🎉
echo ================================================================
echo.
echo 📁 Check the images/ directory for generated visualization files:
echo    • High-quality plots saved at 300 DPI resolution
echo    • Professional styling with emojis and annotations
echo    • Comprehensive analysis dashboards
echo    • Organized folder structure for easy navigation
echo.
echo 💡 Key Features Analyzed:
echo    • Consumption prediction (kWh/100km)
echo    • Energy quantity prediction (kWh)
echo    • ECR deviation analysis
echo    • Individual graphs for detailed analysis
echo    • Combined dashboards for overview
echo.
echo 📂 Folder Structure:
echo    • data_exploration/ - Main dashboard
echo    • individual_graphs/ - Detailed individual plots
echo    • training_analysis/ - Training performance overview
echo    • model_performance/ - Model comparison overview
echo.
echo Press any key to exit...
pause
