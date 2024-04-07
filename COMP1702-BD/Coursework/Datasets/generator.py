import pandas as pd
import numpy as np
from random import choice, uniform
from datetime import datetime, timedelta

# Function to generate a random date within a given range
def random_date(start, end):
    # Returns a date as a datetime object between 'start' and 'end'
    return start + timedelta(days=np.random.randint(0, int((end - start).days)))

# Dictionary of names organized by cultural backgrounds
names = {
    'ME': {  # Middle East
        'first_names': ['Amir', 'Fatima', 'Omar', 'Aisha', 'Khalid', 'Layla', 'Ali', 'Salma', 'Tariq', 'Zara'],
        'last_names': ['Almasi', 'Said', 'Fakhoury', 'Nassar', 'Habib', 'Abadi', 'Maalouf', 'Ganim', 'Antoun', 'Qureshi']
    },
    'IN': {  # Indian
        'first_names': ['Arjun', 'Priya', 'Raj', 'Anjali', 'Sanjay', 'Deepika', 'Amit', 'Neha', 'Vijay', 'Rani'],
        'last_names': ['Sharma', 'Patel', 'Singh', 'Gupta', 'Kumar', 'Joshi', 'Iyer', 'Mehta', 'Desai', 'Chatterjee']
    },
    'US': {  # American
        'first_names': ['Olivia', 'Noah', 'Emma', 'Liam', 'Ava', 'William', 'Sophia', 'James', 'Isabella', 'Benjamin'],
        'last_names': ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Hernandez', 'Lopez']
    }
}

# List of email domains
email_domains = ['gmail.com', 'outlook.com', 'yahoo.com', 'sky.net', 'greenwich.info']

# Function to generate customer data ensuring cultural consistency
def generate_customers():
    customers_list = []
    for customer_id in range(1, 51):  # Ensures unique CustomerID for each entry
        culture_key = choice(list(names.keys()))  # Randomly selects a cultural background
        first_name = choice(names[culture_key]['first_names'])
        last_name = choice(names[culture_key]['last_names'])
        # Constructs an email using the chosen first and last name and a random domain
        email = f"{first_name.lower()}.{last_name.lower()}@{choice(email_domains)}"
        # Generates a random join date formatted as YYYY-MM-DD
        join_date = random_date(datetime(2010, 1, 1), datetime(2022, 1, 1)).strftime('%Y-%m-%d')
        age = np.random.randint(18, 70)  # Random age between 18 and 70
        # Appends a list representing a customer to the customers list
        customers_list.append([customer_id, first_name, last_name, age, email, join_date])
    # Converts the customers list to a DataFrame
    return pd.DataFrame(customers_list, columns=['CustomerID', 'FirstName', 'LastName', 'Age', 'Email', 'JoinDate'])

# Generates the customer DataFrame
customers_df = generate_customers()

# Function to generate product data with categories
def generate_products():
    product_list = []
    product_mapping = {
        'Electronics': ['Laptop', 'Headphones', 'Camera', 'SmartWatch', 'e-Reader'],
        'Clothing': ['Shirt', 'Shoes', 'Watch', 'Jacket', 'Jeans'],
        'Home': ['Sofa', 'Chair', 'Lamp', 'Rug'],
        'Garden': ['Wheelbarrow', 'GardenTools', 'Grill', 'PatioSet'],
        'Toys': ['Game', 'WaterGun', 'Doll', 'Puzzle', 'ActionFigure'],
        'Books': ['Book', 'SmartBook', 'AdvancedBook', 'KidsBook'],
    }

    for product_id in range(101, 151):  # Ensures unique ProductID for each entry
        category = choice(list(product_mapping.keys()))  # Selects a category
        product_type = choice(product_mapping[category])  # Selects a product type within that category
        product_name = product_type
        price = round(uniform(10.0, 500.0), 2)  # Random price between $10 and $500
        supplier_id = np.random.randint(100, 200)  # Random supplier ID between 100 and 200
        # Append data with columns in the correct order as per HiveQL table structure
        product_list.append([product_id, product_name, price, category, supplier_id])
    
    # Create DataFrame with specified column order
    return pd.DataFrame(product_list, columns=['ProductID', 'ProductName', 'Price', 'Category', 'SupplierID'])

# Generates the product DataFrame and saves to CSV
products_df = generate_products()

# Generates sales data with randomized values
sales_data = {
    'SaleID': np.arange(201, 251),  # Sequential SaleID from 201 to 250
    'ProductID': np.random.randint(101, 151, size=50),  # Random ProductID matching the product range
    'CustomerID': np.random.randint(1, 51, size=50),  # Random CustomerID matching the customer range
    'Quantity': np.random.randint(1, 10, size=50),  # Random quantity of product sold between 1 and 10
    'SaleDate': [random_date(datetime(2020, 1, 1), datetime(2022, 1, 1)).strftime('%Y-%m-%d') for _ in range(50)]  # Random sale date
}
# Converts the sales data to a DataFrame
sales_df = pd.DataFrame(sales_data)

# File paths for the CSV files
customers_file_path = 'Customers.csv'
products_file_path = 'Products.csv'
sales_file_path = 'Sales.csv'

# Saves the DataFrames to CSV files
customers_df.to_csv(customers_file_path, index=False)
products_df['ProductID'] = np.arange(101, 151)  # Assigns ProductID after DataFrame creation
products_df.to_csv(products_file_path, index=False)
sales_df.to_csv(sales_file_path, index=False)

# Confirmation message after saving the files
print(f'Files have been saved successfully: {customers_file_path}, {products_file_path}, {sales_file_path}')