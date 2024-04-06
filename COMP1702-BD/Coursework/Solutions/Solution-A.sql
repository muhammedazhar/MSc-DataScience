CREATE DATABASE IF NOT EXISTS `Task-A`;

USE `Task-A`;

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