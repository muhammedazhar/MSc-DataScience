import pandas as pd
import numpy as np
from random import choice, uniform
from datetime import datetime, timedelta

def random_date(start, end):
    return start + timedelta(days=np.random.randint(0, int((end - start).days)))

names = {
    'ME': {
        'first_names': ['Amir', 'Fatima', 'Omar', 'Aisha', 'Khalid', 'Layla', 'Ali', 'Salma', 'Tariq', 'Zara'],
        'last_names': ['Almasi', 'Said', 'Fakhoury', 'Nassar', 'Habib', 'Abadi', 'Maalouf', 'Ganim', 'Antoun', 'Qureshi']
    },
    'IN': {
        'first_names': ['Arjun', 'Priya', 'Raj', 'Anjali', 'Sanjay', 'Deepika', 'Amit', 'Neha', 'Vijay', 'Rani'],
        'last_names': ['Sharma', 'Patel', 'Singh', 'Gupta', 'Kumar', 'Joshi', 'Iyer', 'Mehta', 'Desai', 'Chatterjee']
    },
    'US': {
        'first_names': ['Olivia', 'Noah', 'Emma', 'Liam', 'Ava', 'William', 'Sophia', 'James', 'Isabella', 'Benjamin'],
        'last_names': ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Hernandez', 'Lopez']
    }
}

email_domains = ['gmail.com', 'outlook.com', 'yahoo.com', 'sky.net', 'greenwich.info']

def generate_customers():
    customers_list = []
    for customer_id in range(1, 51):
        culture_key = choice(list(names.keys()))
        first_name = choice(names[culture_key]['first_names'])
        last_name = choice(names[culture_key]['last_names'])
        email = f"{first_name.lower()}.{last_name.lower()}@{choice(email_domains)}"
        join_date = random_date(datetime(2010, 1, 1), datetime(2022, 1, 1)).strftime('%Y-%m-%d')
        age = np.random.randint(18, 70)
        customers_list.append([customer_id, first_name, last_name, age, email, join_date])
    return pd.DataFrame(customers_list, columns=['CustomerID', 'FirstName', 'LastName', 'Age', 'Email', 'JoinDate'])

customers_df = generate_customers()

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
    
    # Generate ProductID in the desired order
    product_ids = np.arange(101, 151)
    
    for product_id in product_ids:
        category = choice(list(product_mapping.keys()))
        product_name, price = choice(product_mapping[category]), round(uniform(10.0, 500.0), 2)
        supplier_id = np.random.randint(100, 200)
        product_list.append([product_id, product_name, price, category, supplier_id])
        
    return pd.DataFrame(product_list, columns=['ProductID', 'ProductName', 'Price', 'Category', 'SupplierID'])

products_df = generate_products()

sales_data = {
    'SaleID': np.arange(201, 251),
    'ProductID': np.random.randint(101, 151, size=50),
    'CustomerID': np.random.randint(1, 51, size=50),
    'Quantity': np.random.randint(1, 10, size=50),
    'SaleDate': [random_date(datetime(2020, 1, 1), datetime(2022, 1, 1)).strftime('%Y-%m-%d') for _ in range(50)]
}

sales_df = pd.DataFrame(sales_data)

customers_df.to_csv('Customers.csv', index=False)
products_df.to_csv('Products.csv', index=False)
sales_df.to_csv('Sales.csv', index=False)

print('Files have been saved successfully.')