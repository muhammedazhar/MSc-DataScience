import pandas as pd
from random import randint, choice, uniform
from datetime import datetime, timedelta

# Function to generate a random date
def random_date(start, end):
    return (start + timedelta(days=randint(0, int((end - start).days))))

# Generate data for Customers table
customers_data = {
    'CustomerID': [i for i in range(1, 51)],
    'FirstName': [f'FirstName{i}' for i in range(1, 51)],
    'LastName': [f'LastName{i}' for i in range(1, 51)],
    'Age': [randint(18, 70) for _ in range(50)],
    'Email': [f'customer{i}@example.com' for i in range(1, 51)],
    'JoinDate': [random_date(datetime(2010, 1, 1), datetime(2022, 1, 1)).date() for _ in range(50)]
}
customers_df = pd.DataFrame(customers_data)

# Generate data for Products table
categories = ['Electronics', 'Clothing', 'Home', 'Garden', 'Toys', 'Books']
products_data = {
    'ProductID': [i for i in range(1, 51)],
    'ProductName': [f'Product{i}' for i in range(1, 51)],
    'Price': [round(uniform(10.0, 500.0), 2) for _ in range(50)],
    'Category': [choice(categories) for _ in range(50)],
    'SupplierID': [randint(100, 200) for _ in range(50)]
}
products_df = pd.DataFrame(products_data)

# Generate data for Sales table
sales_data = {
    'SaleID': [i for i in range(1, 51)],
    'ProductID': [randint(1, 50) for _ in range(50)],
    'CustomerID': [randint(1, 50) for _ in range(50)],
    'Quantity': [randint(1, 10) for _ in range(50)],
    'SaleDate': [random_date(datetime(2020, 1, 1), datetime(2022, 1, 1)).date() for _ in range(50)]
}
sales_df = pd.DataFrame(sales_data)

# Save the dataframes to CSV files
customers_file_path = './customers.csv'
products_file_path = './products.csv'
sales_file_path = './sales.csv'

customers_df.to_csv(customers_file_path, index=False)
products_df.to_csv(products_file_path, index=False)
sales_df.to_csv(sales_file_path, index=False)

(customers_file_path, products_file_path, sales_file_path)