# Task A: Hive Data Warehouse Design

## Overview
Task A involves designing a data warehouse in Hive using original data. This README.md provides a detailed guide to completing Task A based on a tutorial video provided by the course instructor.

## Coursework Requirements
- **Data Creation**: Create a data warehouse with a minimum of three tables and at least 50 records in total.
- **Query Design**: Develop 10 different queries demonstrating complexity and variety.
- **Screenshots**: Include screenshots for each query and its results to validate implementation.

## Tutorial Video Summary
The [tutorial video](https://youtu.be/ojdtg9z3Xro?si=xYz4ONPYmwErt2nk) offers insights into Task A, emphasizing the following key points:

### Data Creation
- **Original Data**: Building the data warehouse with original data is crucial for demonstrating your understanding of data manipulation and analysis. Consider generating data that aligns with your coursework topic or interests, ensuring it's diverse and representative.
- **Table Requirements**: Creating at least three tables with more than 50 records collectively offers ample opportunity to showcase your database design skills. Aim for a balance between the number of tables and the complexity of their relationships.
- **CSV File Building**: Whether you choose to create CSV files directly on the virtual machine or copy them from your laptop, ensure the data is formatted correctly. Pay attention to data types, delimiter usage, and consistency to avoid errors during the upload process.

### Hive Setup and Verification
- **Virtual Machine Access**: Accessing the assigned virtual machine is fundamental to completing the task. Follow the instructions provided by your instructor or institution to ensure seamless access without any disruptions.
- **Hive Environment**: Before proceeding with data upload and query execution, verify that Hive is functioning correctly. Test basic commands to ensure connectivity and functionality within the Hive environment.
- **Hive Service Management**: Familiarize yourself with commands like `start-dfs.sh` or `hadoop namenode -format` for managing the Hive service. Understanding these commands will empower you to troubleshoot any issues that may arise during the process.

### Data Upload and Table Creation
- **CSV Upload**: Use the `hdfs dfs -put` command to upload CSV files into Hive. Double-check the file paths and permissions to ensure successful upload without any data loss or corruption.
- **Table Creation**: When designing tables within the Hive environment, consider factors such as data types, column names, and primary keys. Use the `CREATE TABLE` command judiciously to define table structures that align with your data requirements.
- **Data Loading**: Once tables are created, use the `LOAD DATA` command to populate them with the uploaded data. Verify the integrity of the loaded data by cross-referencing it with the original CSV files.

### Query Design and Execution
- **Complex Query Requirements**: Design queries that go beyond simple data retrieval, showcasing your ability to perform complex operations such as joins, aggregations, and subqueries. Incorporate multiple tables to demonstrate proficiency in handling relational data.
- **Screenshot Inclusion**: Capture screenshots of each query along with its corresponding results to provide visual evidence of your query execution. Ensure that the screenshots are clear and well-annotated to facilitate understanding and evaluation.
- **Table Record Verification**: Display screenshots of each table, highlighting the record count to confirm compliance with the requirement of more than 50 records per table. Include additional details such as table structure and sample records to provide comprehensive documentation.

## Conclusion
This README.md serves as a comprehensive guide for completing Task A, providing step-by-step instructions based on the tutorial video's content. Adherence to the outlined process will ensure successful completion of the coursework requirements and the attainment of a comprehensive understanding of Hive data warehousing principles.