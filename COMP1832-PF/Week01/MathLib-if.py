# Lecture 1: Math Library with if statement
import math

num = float(input("Enter a number: "))
if num >= 0:
    print(math.sqrt(num))
else:
    print("Cannot compute the square root of a negative number")
# This program computes the square root of a number, but only if it is positive.