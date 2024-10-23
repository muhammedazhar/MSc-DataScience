# Lab 1: Introduction to R

# Exercise 1: Print the following message to the console
print("Welcome to Time Series Analysis!")

# Exercise 2: Create two variables a and b, then list them
a <- 10
b <- 20
ls()  # List all variables in the environment

# Exercise 3: Create two vectors v1 and v2
v1 <- c(1, 2, 3, 4, 5)
v2 <- c(10, 9, 8, 7, 6)

# Exercise 4: Create a sequence from 1 to 20 with step size 2
seq1 <- seq(1, 20, by = 2)
seq1  # Display the sequence

# Exercise 5: Compare vectors v1 and v2
v1 > v2  # Element-wise comparison

# Exercise 6: Select the 3rd element of v1 and the 2nd to 4th elements of v2
v1[3]       # 3rd element of v1
v2[2:4]     # 2nd to 4th elements of v2

# Exercise 7: Perform element-wise addition of v1 and v2
v1 + v2

# Exercise 8: Evaluate the expression (5 + 3) * 2 - 4^2
result <- (5 + 3) * 2 - 4^2
result  # Display the result

# Exercise 9: Create a data frame with the specified structure
df <- data.frame(
  Name = c("John", "Jane", "Alex"),
  Age = c(25, 30, 28),
  Salary = c(40000, 50000, 45000)
)
df  # Display the data frame
