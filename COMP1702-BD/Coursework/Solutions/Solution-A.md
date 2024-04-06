# Task A: Hive Data Warehouse Design

## Objective
Design a Hive-based data warehouse with at least 50 records spanning at least three tables. Implement and document ten varied and complex queries demonstrating the warehouse's functionality.

## Setup and Data Preparation

### Accessing Hadoop Virtual Machine

- Access the Hadoop Virtual Machine as per Week 2's lab document.
- For home access, use the student virtual desktop at [https://rdweb.wvd.microsoft.com/arm/webclient/index.html](https://rdweb.wvd.microsoft.com/arm/webclient/index.html), then access the Hadoop VM.

### Data Preparation

1. **Build CSV Files**: Create CSV files for your Hive tables. Ensure your dataset contains at least 50 records distributed across at least three tables.

2. **Transfer Data**: Due to restrictions, transfer data via copy/paste directly to a file in the Hadoop VM or edit files directly on the VM.

### Starting Hadoop Services

- Initiate Hadoop with `start-all.sh`. If issues arise, restart Hadoop with `stop-all.sh` followed by `start-all.sh`.

### Uploading Data to Hadoop

- Upload your CSV files to Hadoop using `hdfs dfs -put <filename>.csv`.

## Hive Data Warehouse Implementation

### Accessing Hive

- Enter the Hive interface with the command: `hive`.

### Creating Tables

```sql
CREATE TABLE example_table (
    column1 STRING, 
    column2 STRING, 
    column3 INT
) ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ',' 
STORED AS TEXTFILE;
```

- Repeat this process to create at least three tables with your dataset's structure.

### Loading Data into Hive Tables

```sql
LOAD DATA INPATH 'path/to/your/file.csv' 
OVERWRITE INTO TABLE example_table;
```

- Ensure your data is correctly placed in Hadoop before this step.

## Query Implementation and Execution

### Sample Queries

Here are examples of the types of queries you could run:

1. **Query Example**: Count records in a table.

```sql
SELECT COUNT(*) FROM example_table;
```

2. **Complex Query Example**: Aggregate functions with GROUP BY.

```sql
SELECT column1, AVG(column3) 
FROM example_table 
GROUP BY column1;
```

- Document and execute a total of ten queries showcasing a variety of operations such as joins, aggregations, and subqueries.

## Documentation and Discussion

- **Explanation**: For each query, provide a brief explanation of its purpose and the insights it offers about the data.
- **Screenshots**: Include screenshots of the query execution and results within Hive as proof of your implementation.

## Conclusion

This document outlines the process for designing and querying a data warehouse in Hive, demonstrating a variety of techniques to manipulate and extract insights from big data.