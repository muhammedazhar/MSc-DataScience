You are to carry out a visual data exploration for ChrisCo, the fictional company whose sales and website data we have been analysing throughout the module. 

The ChrisCo company is fictional but nonetheless very successful. It manages a chain of cinemas across the UK. ChrisCo collects a vast amount of data about individual customers visiting its cinemas using its loyalty card scheme, and this customer data has been aggregated/averaged over a four-year period to give information about the company’s cinemas, each identified by a unique 3-letter code (e.g., ABC, XYZ, etc.). 

You should compile the data into two data frames: one containing customer data (one row for each date) and the other containing summary data (one row for each cinema), compiled from all of the .csv files, including the visitor data.

Your task is to investigate the data visually and present some conclusions about any characteristics you discover, including correlations, seasonal behaviour, outliers, etc., together with a suggestion about how the data might be best segmented based on the total volume of visits at each cinema.

The company is most interested in the high and medium volume cinemas but would like a summary of the low volume cinemas plus any anomalies you identify in the data. You should also identify new cinemas that have been opened over the period or cinemas that the company has closed over the period.

You should present your findings in the form of data visualisations for the company, i.e. based on the assumption that the reader knows nothing about data visualisation.

The code should make use of some of the examples taught in the lectures. You may use additional code examples from elsewhere but only to supplement the lecture examples, not to replace them entirely. At least 2 of the visualisations in the notebook should be interactive and provide functionality to explore the data in more detail.

Here is the dataset links as follows;

https://tinyurl.com/ChrisCoDV/001364857/CinemaWeeklyVisitors.csv
https://tinyurl.com/ChrisCoDV/001364857/CinemaAge.csv
https://tinyurl.com/ChrisCoDV/001364857/CinemaCapacity.csv
https://tinyurl.com/ChrisCoDV/001364857/CinemaMarketing.csv
https://tinyurl.com/ChrisCoDV/001364857/CinemaOverheads.csv
https://tinyurl.com/ChrisCoDV/001364857/CinemaSpend.csv

Note: You don't have to download the datasets on your own. You can view it by using `pd.read_csv(link)`.

1. Time Series Analysis (Line Chart): To observe the seasonal behavior and trends over time in cinema visits. This can highlight peak seasons, possibly correlating with major holiday periods or blockbuster releases.

2. Cinema Capacity vs. Average Weekly Visitors (Bubble Chart): This visualization can help identify the correlation between cinema capacity and the volume of visitors. Bubble size will represent the total volume of visits, providing immediate visual cues for high, medium, and low volume cinemas.

3. Cinema Age and Visitor Volume (Scatter Plot): A scatter plot to explore if there's a relationship between how long a cinema has been operational (age) and the average number of visitors. This might reveal trends such as newer cinemas attracting more visitors or vice versa.

4. Marketing Spend vs. Visitor Volume (Scatter Plot with Trend Line): To check for correlations between the amount spent on marketing by each cinema and the number of visitors. This could help in assessing the effectiveness of marketing strategies.

5. Cinema Overheads vs. Visitor Volume (Horizontal Bar Chart): A comparison of overhead costs with the visitor volume to identify if higher spending cinemas are necessarily attracting more visitors. This can also highlight anomalies where overheads are high but not matched by visitor numbers.

6. Visitor Volume Segmentation (Treemap): Segmenting cinemas based on the total volume of visits into high, medium, and low categories. This visualization will provide a clear overview of how many cinemas fall into each segment and allow for easy identification of outliers.

7. Interactive Cinema Performance Dashboard (Interactive Area Chart): An interactive dashboard that allows users to select specific cinemas or time periods to examine trends in visitor numbers, spending, and other relevant metrics in detail.

8. Cinema Lifecycle (Stacked Bar Chart): Showing new openings and closures of cinemas over the period. Each bar can represent a year with segments showing the number of cinemas opened or closed, giving insights into the growth or contraction of the chain.

Before that, here is the code I have already run according to the task. Can you please check and verify this? Moreover, I want you to rewrite your code to match my variables. You also note that every time the code needs to create a visualisation, it should NOT manipulate the main data frames. Instead, it should create a copy of the main data frame for manipulation.

You should utilize the compiled data (customer_df and summary_df): one containing customer data (one row for each date) and the other containing summary data (one row for each cinema), compiled from all of the .csv files, including the visitor data.