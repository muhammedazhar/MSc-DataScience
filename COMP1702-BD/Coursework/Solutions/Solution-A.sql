CREATE DATABASE IF NOT EXISTS `Task_A`;

USE `Task_A`;

-- Table: Customers
CREATE TABLE Customers (
    CustomerID INT,
    FirstName STRING,
    LastName STRING,
    Age INT,
    Email STRING,
    JoinDate DATE
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE;

LOAD DATA INPATH 'Customers.csv' OVERWRITE INTO TABLE Customers;

-- Table: Products
CREATE TABLE Products (
    ProductID INT,
    ProductName STRING,
    Price DECIMAL(10,2),
    Category STRING,
    SupplierID INT
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE;

LOAD DATA INPATH 'Products.csv' OVERWRITE INTO TABLE Products;

-- Table: Sales
CREATE TABLE Sales (
    SaleID INT,
    ProductID INT,
    CustomerID INT,
    Quantity INT,
    SaleDate DATE
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE;

LOAD DATA INPATH 'Sales.csv' OVERWRITE INTO TABLE Sales;

-- Query 1: List all customers over 30 years old.
SELECT * FROM Customers WHERE Age > 30;

-- Query 2: Find the average price of products in each category.
SELECT Category, AVG(Price) as AvgPrice FROM Products GROUP BY Category;

-- Query 3: Show total sales by product name.
SELECT P.ProductName, SUM(S.Quantity) as TotalUnitsSold
FROM Sales S
JOIN Products P ON S.ProductID = P.ProductID
GROUP BY P.ProductName;

-- Query 4: Identify the most recent purchase of each customer.
SELECT CustomerID, MAX(SaleDate) as MostRecentPurchase
FROM Sales
GROUP BY CustomerID;

-- Query 5: Calculate the total revenue per category.
SELECT P.Category, SUM(P.Price * S.Quantity) as TotalRevenue
FROM Sales S
JOIN Products P ON S.ProductID = P.ProductID
GROUP BY P.Category;

-- Query 6: List products that have never been sold.
SELECT p.*
FROM Products p
LEFT JOIN (SELECT DISTINCT ProductID FROM Sales) s ON p.ProductID = s.ProductID
WHERE s.ProductID IS NULL;

-- Query 7: Determine the number of customers who joined each month.
SELECT MONTH(JoinDate) as JoinMonth, COUNT(*) as CustomerCount
FROM Customers
GROUP BY MONTH(JoinDate);

-- Query 8: Show top 5 customers by total spend.
SELECT C.CustomerID, SUM(P.Price * S.Quantity) as TotalSpend
FROM Sales S
JOIN Products P ON S.ProductID = P.ProductID
JOIN Customers C ON S.CustomerID = C.CustomerID
GROUP BY C.CustomerID
ORDER BY TotalSpend DESC
LIMIT 5;

-- Query 9: Count the number of products in each category with a price above $50.
SELECT Category, COUNT(*) as ProductCount
FROM Products
WHERE Price > 50
GROUP BY Category;

-- Query 10: Show the monthly sales trend for a specific product.
SELECT MONTH(SaleDate) as SaleMonth, COUNT(*) as NumberOfSales
FROM Sales
WHERE ProductID = 101
GROUP BY MONTH(SaleDate);