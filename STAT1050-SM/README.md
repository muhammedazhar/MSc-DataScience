# STAT1050 - Statistical Methods for Time Series Analysis

**Institution**: University of Greenwich  
**Faculty**: Faculty of Engineering and Science  
**School**: School of Computing and Mathematical Sciences  
**Credits**: 15  
**Module Delivery:** Term 1  
**Module Leader:** Dr. Konstantinos Skindilias  
**Module Instructor:** Dr. Richard Quibell  

## Overview

The *Statistical Methods for Time Series Analysis (STAT1050)* module provides foundational knowledge of statistical techniques to analyze time series data, crucial for computer science students. Time series data, characterized by observations recorded sequentially over time, introduces unique challenges in statistical modeling and inference. This module aims to equip students with the skills to apply various statistical techniques, model time-dependent data, and understand the implications of their analyses in fields such as quantitative finance and automated securities trading.

The module is intended to provide practical skills in analyzing real-world time series problems, contributing to better decision-making through statistical modeling. Time series analysis plays a vital role in numerous fields, including finance, economics, environmental studies, and engineering, where data is inherently time-ordered.

## Learning Outcomes

Upon successful completion of the module, students will be able to:

- Critically evaluate time series models and their assumptions.
- Utilize appropriate statistical and Machine Learning tools for time series forecasting.
- Understand AR, MA, and ARIMA models and their applications in time series analysis.

## Module Content

The module introduces students to the following key topics:

1. **Evaluate Model Outputs and Assumptions**  
   Critically evaluate statistical model outputs and underlying model assumptions, understanding their implications for advanced modeling and inference in time series analysis.

2. **Apply Statistical Tools for Analysis**  
   Demonstrate the ability to make informed choices about the appropriate statistical tools for time series analysis, model fitting, and forecasting. Students will also be able to apply linear regression models, select and evaluate alternative models, and understand the contexts where Machine Learning techniques are useful.

3. **Understand AR, MA, and ARIMA Models**  
   Demonstrate a comprehensive understanding of autoregressive (AR) and moving average (MA) processes, as well as their coupling as ARIMA (Autoregressive Integrated Moving Average) processes. Students will apply these models to time series data using statistical packages and, where appropriate, Machine Learning techniques, and critically interpret and report on the results.

### Time Series Data

Time series data refers to data points collected or recorded at successive time intervals. Analysis of such data requires techniques that account for temporal dependencies, which are unique compared to other types of statistical data.

### Statistical Techniques for Time Series

1. **Autoregressive (AR) Models**: These models predict future values based on past values of the series. The AR model structure makes it useful for capturing dependencies in time series data.

2. **Moving Average (MA) Models**: MA models express a variable as a linear function of current and past disturbances (error terms), capturing the shock or randomness over time.

3. **Autoregressive Integrated Moving Average (ARIMA) Models**: ARIMA is a generalization of AR and MA models used for modeling time series data. It accounts for differencing to make non-stationary data stationary and is widely applied for both estimation and forecasting.

4. **Seasonal Models**: These models address time series data with seasonal patterns, allowing for improved forecasting accuracy in applications such as demand forecasting.

### Software Tools

The module makes extensive use of **R**, a powerful programming language for statistical computing. R's packages for time series analysis, such as `forecast`, `tseries`, and others, are used to implement the statistical techniques taught.

## Schedule of Teaching and Learning Activities

The module is delivered over the course of 12 weeks, including lectures, tutorials, and a personal development week. Below is an outline of the main topics covered each week:

| Week | Date (Week Beginning) | Activity                                                           |
|------|-----------------------|-------------------------------------------------------------------|
| 1    | 23/09/2024            | Session 1 – Intro to the module & key R modules                    |
| 2    | 30/09/2024            | Session 2 – Key probability concepts                               |
| 3    | 07/10/2024            | Session 3 – General Statistics                                     |
| 4    | 14/10/2024            | Session 4 – Time Series Basics > Sample ACF and Properties of AR(1) Model |
| 5    | 21/10/2024            | Session 5 – Moving Average Models (MA models) and Partial Autocorrelation Function (PACF) |
| 6    | 28/10/2024            | Session 6 – ARIMA models estimation and diagnostics                |
| 7    | 04/11/2024            | Skills Week (No Lectures & Tutorials)                            |
| 8    | 11/11/2024            | Session 8 – ARIMA models forecasting                               |
| 9    | 18/11/2024            | Session 9 – Seasonal models                                        |
| 10   | 25/11/2024            | Session 10 – ARIMA vs Machine Learning (ML) models estimation and diagnostics |
| 11   | 02/12/2024            | Session 11 – ARIMA vs ML comparison / Coursework submission       |
| 12   | 09/12/2024            | Session 12 – ARIMA vs ML models and Value-at-Risk comparison      |

### Teaching Approach

- **Lectures**: Weekly 2-hour lectures focusing on theory and introducing the tools and techniques required for time series analysis.
- **Tutorials and Group Study**: One-hour lab sessions where students practice implementing the techniques in R, and group study sessions for collaborative problem-solving.
- **Self-Directed Learning**: Students are expected to independently review the lecture material, conduct further research on open-source libraries, and work on their coursework.

## Assessment

### Assessment Structure

The module's assessment is based solely on coursework, contributing 100% of the final grade. The coursework requires students to apply the learned statistical techniques to a dataset, analyze the time series behavior, fit an appropriate model, and make predictions. Students are expected to demonstrate a critical understanding of the model's assumptions and their implications.

- **Submission Deadline**: 6th December 2024 (23:30 UK time).
- **Length**: Maximum of 4 pages.

### Learning Outcomes for Assessment

The coursework is designed to evaluate students' ability to:

- Develop and assess time series models.
- Make data-driven decisions using appropriate statistical and Machine Learning tools.
- Interpret and report the implications of statistical analyses for decision-making.

## Reading Resources

The following resources are suggested to support learning in the module:

1. **An Introduction to R**: A comprehensive guide to R programming, covering structure, visualization, and essential statistical functions.
2. **R Cookbook (2nd Edition)**: A practical guide for using R in the context of data structures, probability theory, statistics, and time series analysis.

## Additional Information

- Lecture Recordings: All lectures will be recorded and made available through the Panopto system for revision purposes.
- Lab Work: It is recommended to complete any unfinished lab exercises independently to maximize understanding.

The module aims to provide a comprehensive foundation in the statistical analysis of time series data, which is essential for understanding and forecasting temporal patterns in various domains, from finance to environmental studies.
