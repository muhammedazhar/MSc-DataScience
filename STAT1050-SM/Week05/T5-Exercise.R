# Time Series Analysis: MA Models and Stationarity Tests
# Source: https://www.rdocumentation.org/packages/TimeSeries.OBeu/versions/1.2.4/topics/ts.stationary.test

## 1. Simulating an MA(3) Process and Plotting Against a Normal Distribution

# In this exercise, we will simulate an MA(3) process and compare it to a 
# normal distribution.

### Load necessary libraries
if (!require(forecast)) install.packages("forecast", dependencies = TRUE)
library(forecast)

### Set the seed for reproducibility
set.seed(123)

### Simulate an MA(3) process with specific parameters
ma_coeffs <- c(0.5, -0.3, 0.2)  # Coefficients for the MA process
n <- 1000  # Number of observations
ma_process <- arima.sim(n = n, list(ma = ma_coeffs))

### Plot the simulated MA(3) process
plot(ma_process, type = "l", main = "Simulated MA(3) Process", ylab = "y_t")

### Plot the histogram of the simulated MA(3) process
hist(ma_process, breaks = 30, probability = TRUE,
     main = "Histogram of Simulated MA(3) Process", xlab = "Value")

### Overlay a normal distribution on the histogram for comparison
lines(density(ma_process), col = "red", lwd = 2)  # Kernel density estimate
curve(dnorm(x, mean(ma_process), sd(ma_process)), add = TRUE, 
      col = "blue", lwd = 2)  # Normal distribution
legend("topright",
       legend = c("Density of MA(3)", "Normal Distribution"),
       col = c("red", "blue"), lwd = 2)

## 2. Stationarity Tests

### We know certain processes:
# - White Noise (WN) process: stationary by default
# - Random Walk (RW): non-stationary
# - Differencing a RW: makes it stationary
# - Adding a trend to WN: makes it trend-stationary

### Create time series data for stationarity tests
t <- 0:300
y_stationary <- rnorm(length(t), mean = 1, sd = 1)  # Stationary time series
y_trend <- cumsum(rnorm(length(t), mean = 1, sd = 4)) + t / 100  # Trend TS

### Normalize the series for simplicity
y_stationary <- y_stationary / max(y_stationary)
y_trend <- y_trend / max(y_trend)

### Visualize the signals and their ACFs
par(mfcol = c(2, 2))  # Set up a 2x2 plotting area

# Plot the stationary signal and ACF
plot(t, y_stationary, type = "l", col = "red", 
     xlab = "Time (t)", ylab = "Y(t)", main = "Stationary Signal")
acf(y_stationary, lag.max = length(y_stationary), main = "ACF of Stationary Signal")

# Plot the trend signal and ACF
plot(t, y_trend, type = "l", col = "red", 
     xlab = "Time (t)", ylab = "Y(t)", main = "Trend Signal")
acf(y_trend, lag.max = length(y_trend), main = "ACF of Trend Signal")

### Observations:
# - Stationary signal: ACF lags die out quickly.
# - Trend signal: ACF lags persist, indicating non-stationarity.

## 3. Stationarity Tests Using Augmented Dickey–Fuller (ADF) and KPSS Tests

# Load the necessary library for stationarity tests
if (!require(tseries)) install.packages("tseries", dependencies = TRUE)
library(tseries)

### ADF Test: Test for a unit root (non-stationarity)
# y_stationary tests
adf.test(y_stationary)  # Default test
adf.test(y_stationary, alternative = "stationary", k = trunc((length(y_stationary)-1)^(1/3)))  # Alternative hypothesis: stationary
adf.test(y_stationary, alternative = "explosive", k = trunc((length(y_stationary)-1)^(1/3)))  # Alternative hypothesis: explosive

### KPSS Test: Null hypothesis of stationarity
# Test for level and trend stationarity
kpss.test(y_stationary, null = "Level")  # Test for level stationarity
kpss.test(y_stationary, null = "Trend")  # Test for trend stationarity

# y_trend tests
adf.test(y_trend, alternative = "stationary")  # Test for stationary alternative
adf.test(y_trend, alternative = "explosive")  # Test for explosive alternative
adf.test(y_trend)  # Default test

### KPSS Test on y_trend
kpss.test(y_trend, null = "Level")  # Test for level stationarity
kpss.test(y_trend, null = "Trend")  # Test for trend stationarity
