CREATE DATABASE IF NOT EXISTS `Task-A` DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;

-- Table: Customers
CREATE TABLE Customers (
    CustomerID INT,
    FirstName STRING,
    LastName STRING,
    Age INT,
    Email STRING,
    JoinDate DATE
) STORED AS TEXTFILE;

-- Table: Products
CREATE TABLE Products (
    ProductID INT,
    ProductName STRING,
    Price DECIMAL(10,2),
    Category STRING,
    SupplierID INT
) STORED AS TEXTFILE;

-- Table: Sales
CREATE TABLE Sales (
    SaleID INT,
    ProductID INT,
    CustomerID INT,
    Quantity INT,
    SaleDate DATE
) STORED AS TEXTFILE;