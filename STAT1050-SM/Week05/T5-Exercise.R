# https://www.rdocumentation.org/packages/TimeSeries.OBeu/versions/1.2.4/topics/ts.stationary.test


# Time Series Analysis: MA Models and Stationarity Tests

## 1. Simulating an MA(3) Process and Plotting Against a Normal Distribution
# In this exercise, we will simulate an MA(3) process and compare it to a 
# normal distribution.

### R Code:

# Load the necessary libraries
if (!require(forecast)) install.packages("forecast")
library(forecast)

# Set the seed for reproducibility
set.seed(123)

# Simulate an MA(3) process with specific parameters
ma_coeffs <- c(0.5, -0.3, 0.2)  # Coefficients for the MA process
n <- 1000  # Number of observations
ma_process <- arima.sim(n = n, list(ma = ma_coeffs))

# Plot the simulated MA(3) process
plot(ma_process, type = "l", main = "Simulated MA(3) Process", ylab = "y_t")

# Plot the histogram of the simulated MA(3) process
hist(ma_process, breaks = 30,
     probability = TRUE,
     main = "Histogram of Simulated MA(3) Process", xlab = "Value")

# Overlay a normal distribution on the histogram for comparison
lines(density(ma_process), col = "red", lwd = 2)
curve(dnorm(x, mean(ma_process),
            sd(ma_process)), add = TRUE, col = "blue", lwd = 2)
legend("topright",
       legend = c("Density of MA(3)", "Normal Distribution"),
       col = c("red", "blue"), lwd = 2)



# We want to test for stationarity
# We may find ourselfs in a situation that we are not sure how to properly interpret the test output
# We should first go to R Help (F1) and read the documentation
# Then we can actually create data that we know whether or not are stationary and perform the tests
# For example:
# We know what a White Noise (WN) process is stationary (by default)
# We know that a simple Random Walk (RW) is not stationary
# We know that if we difference a simple RW then the process will become stationary
# We know that if we add to a WN process a trend it will become Trend-Stationary (see below)


# Practice on stationarity by testing on output that you know what to expect
t = 0:300
y_stationary <- rnorm(length(t),mean=1,sd=1) # the stationary time series (ts)
y_trend      <- cumsum(rnorm(length(t),mean=1,sd=4))+t/100 # our ts with a trend
# lets normalize each for simplicity (do you know what normalisation does)
y_stationary<- y_stationary/max(y_stationary)
y_trend      <- y_trend/max(y_trend)

# Second, we can check each for characteristics of stationarity
# by looking at the autocorrelation functions (ACF) of each signal.
# For a stationary signal, because we expect no dependence with time,
# we would expect the ACF to go to 0 for each time lag (τ).
# Lets visualize the signals and ACFs:

plot.new()
frame()
par(mfcol = c(2, 2))
# the stationary signal and ACF
plot(t, y_stationary,
     type = "l", col = "red",
     xlab = "time (t)",
     ylab = "Y(t)",
     main = "Stationary signal")
acf(y_stationary, lag.max = length(y_stationary),
    xlab = "lag #", ylab = "ACF", main = " ")
# the trend signal and ACF
plot(t, y_trend,
     type = "l", col = "red",
     xlab = "time (t)",
     ylab = "Y(t)",
     main = "Trend signal")
acf(y_trend, lag.max = length(y_trend),
    xlab = "lag #", ylab = "ACF", main = " ")

# Notably, the stationary signal (top left) results in few
# significant lags that exceed the confidence interval of
# the ACF (blue dashed line, bottom left) . In comparison,
# the time series with a trend (top right) results in almost
# all lags exceeding the confidence interval of the ACF (bottom right).
# Qualitatively, we can see and conclude from the ACFs that the signal
# on the left is stationary (due to the lags that die out) while the signal
# on the right is not stationary (since later lags exceed the confidence
# interval).

# Test on stationarity
library(tseries)
# A test we can conduct is the Augmented Dickey–Fuller (ADF)
# t-statistic test to find if the series has a unit root
# (a series with a trend line will have a unit root and result in a large p-value).
# y_stationary (check the help function - you can test for different things)
adf.test(y_stationary)
adf.test(y_stationary, alternative = c("stationary"),k = trunc((length(y_stationary)-1)^(1/3)))
adf.test(y_stationary, alternative = c("explosive"),k = trunc((length(y_stationary)-1)^(1/3)))

# KPSS: a null hypothesis that an observable time series is stationary
# We can test if the time series is level or trend
# stationary using the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.
# Here we will test the null hypothesis of trend stationarity
# (a low p-value will indicate a signal that is not trend stationary, has a unit root):
tseries::kpss.test(y_stationary, null = "Level")
tseries::kpss.test(y_stationary)
tseries::kpss.test(y_stationary, null = "Trend")

# y_trend
adf.test(y_trend, alternative = c("stationary"))
adf.test(y_trend, alternative = c("explosive"))
adf.test(y_trend)

tseries::kpss.test(y_trend, null = "Level")
tseries::kpss.test(y_trend)
tseries::kpss.test(y_trend, null = "Trend")
