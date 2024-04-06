# Task B: MapReduce Programming

This document outlines the specifications and requirements for Task B of the COMP1702 coursework, focusing on MapReduce programming. The task aims at processing a computer science bibliography file stored on Hadoop to perform specific data analysis operations, as determined by the student's ID.

## Overview

Task B requires students to utilize MapReduce programming paradigm to analyze a dataset consisting of bibliographic records of computer science papers. Each record follows a predefined format, including the authors, title, conference, and year of publication, separated by the "`|`" character.

### Objective

- **Task B4:** Calculate the average number of authors per paper for each year within the dataset.

## Selection Criteria

Students are assigned specific subtasks based on the last digit of their student ID. It's crucial to select the correct task to avoid penalties. If unsure about the task assignment, students are encouraged to contact the instructor for clarification.

## Implementation Details

### Programming Language

- **Options:** Scala or Java
- **Recommendation:** `Scala` is highly recommended for its concise syntax, especially for students with no Java background.

### Code Structure

- **Mapper and Reducer Classes:** Students must design at least two classes: `Mapper` and `Reducer`.

- **Functionality:** Implement `map` and `reduce` functions to process the input data and produce the desired output.
- **Efficiency:** The algorithm should be optimized for efficiency, utilizing combiners or in-map combiners as appropriate.

#### Mapper and Reducer Classes

For Task B4, students are required to design two primary classes: `Mapper` and `Reducer`, as part of the MapReduce programming model. These classes play a crucial role in processing the dataset and achieving the task objective.

- **Mapper Class:** The `Mapper` class is responsible for reading the input data and transforming it into intermediate key-value pairs. Each line of input data represents a bibliographic record formatted as "authors|title|conference|year". The mapper processes this input to produce a key-value pair where the key is the year of publication and the value is the number of authors of the paper.

  ```java
  class Mapper {
      public void map(String key, String value, Context context) {
          // Extract year and authors from the input string
          // Emit (year, authorsCount) as intermediate key-value pair
      }
  }
  ```

- **Reducer Class:** The `Reducer` class takes the intermediate key-value pairs produced by the mapper(s) and aggregates them to compute the final output. For this task, the reducer will calculate the average number of authors per paper for each year, emitting a key-value pair where the key is the year and the value is the average number of authors.

  ```java
  class Reducer {
      public void reduce(String key, Iterable<Integer> values, Context context) {
          // Calculate the average number of authors per paper for the year
          // Emit (year, averageAuthors) as the final result
      }
  }
  ```

### Input and Output

The map and reduce functions are the core of the MapReduce programming model, determining how input data is transformed into the desired output.

- **Map Function:** The map function processes individual records of the dataset, emitting intermediate key-value pairs for further processing. The key is typically a factor by which to aggregate the data (e.g., publication year), and the value is the datum to be analyzed (e.g., the count of authors).

  - *Input Key-Value Pair:* Represents individual records from the dataset.
  - *Output Key-Value Pair:* Intermediary results to be processed by the reduce function.

- **Reduce Function:** The reduce function aggregates the intermediate data grouped by key, performing calculations or transformations to produce the final result. In this task, it involves computing the average number of authors per paper for each year based on the intermediate data provided by the map function.

  - *Input Key-Value Pair:* Aggregated results from the map function.
  - *Output Key-Value Pair:* Final results, such as the average number of authors per year.


## Efficiency and Optimization

To enhance the algorithm's efficiency, students should explore the use of combiners and in-map combiners, as discussed in the course materials. An efficient solution is expected to handle large datasets with minimal resource consumption.

Efficiency in a MapReduce algorithm is achieved through the effective use of computational resources, minimizing the time and space required to process large datasets. To enhance efficiency:

- **Combiners:** A combiner is an optional class that performs local aggregation of intermediate data on the same node where the map function is executed. It reduces the amount of data transferred across the network for the reduce phase.

- **In-Map Combiners:** An in-map combiner approach involves performing some aggregation directly within the map function, further reducing the volume of intermediate data. This method requires careful management of memory and computation within the mapper.

By optimizing the map and reduce functions and judiciously using combiners, students can significantly improve the performance of their MapReduce tasks, handling large datasets with greater efficiency.

## Submission Guidelines

- **Code:** Submit the implemented solution in Scala or Java, adhering to the specified requirements.
- **Documentation:** Include detailed comments explaining the logic behind the map and reduce functions, key-value pairs used, and any optimization techniques applied.
- **Analysis:** Provide an analysis of the algorithm's efficiency, explaining the choice of combiners or in-map combiners and their impact on performance.

## Resources

- **Lecture Slides:** Review the lecture slides on Advanced MapReduce programming available on the course Moodle page.
- **Additional Readings:** Consult recommended books and resources for deeper insights into MapReduce programming principles and best practices.

There is YouTube video provided by the professor explaining this task. For any uncertainties or questions regarding Task B, students should not hesitate to watch the [YouTube video](https://www.youtube.com/watch?v=d3wYKU7PKn0) or email the instructor directly for guidance.