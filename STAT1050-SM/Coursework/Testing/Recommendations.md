# Recommendations for the coursework

1. Outlier Commentary: In Step 3, you detected 43 potential outliers. Consider adding a small markdown cell to discuss whether these outliers coincide with significant economic events or represent anomalies that require further inspection. This would help to solidify your understanding of why these data points are notable.

2. Exploratory Planning: In Step 5, consider adding a short section on how your observations about volatility, outliers, and trends will impact your approach to ARIMA modeling. This will demonstrate a comprehensive and thoughtful workflow to your grader.

3. Notebook Flow: Your notebook is well-structured with clear separation of each step. The markdown commentary is thorough and clearly supports your decisions. Adding forward-looking statements (e.g., "This observation will help us decide whether differencing is necessary during ARIMA modeling...") will show a clear vision for subsequent steps.

4. Markdown Commentary on Plots: Include a brief analysis of the ACF and PACF plots to show how you decided on initial AR and MA orders. Mention specific lags where significant spikes occurred.
Discuss the difference between log-transformed and differenced plots to demonstrate how each transformation impacts the dataset.

5. `auto.arima()` Interpretation: Provide some commentary on why `auto.arima()` selected the specific model parameters (ARIMA(1, 2, 0)), especially focusing on why d = 2 might have been chosen instead of d = 1.

6. Seasonality Consideration: After plotting extended ACF and PACF, mention if you observed any clear seasonal patterns or if they were insignificant, and explain why you concluded whether or not a SARIMA model was necessary.
