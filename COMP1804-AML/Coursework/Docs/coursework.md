Here's the scenario: You're consulting for a non-profit organization called NotTheRealWikipedia. They're interested in using machine learning to automatically analyze new content added to their site. They've provided a dataset with sample texts and their associated topics, as well as other potentially useful information. The organization is particularly interested in analyzing paragraph-sized texts, as new articles and edits on their website can be short.

They've tasked you with two main objectives:
1. Firstly, they want to determine if machine learning can accurately identify the topic of a paragraph of text, specifically focusing on the topics mentioned earlier. They've also provided a feature indicating whether each paragraph contains references to a person, organization, and/or a product, which they believe could provide additional relevant information.
- 1a. The topic to predict is listed in the column titled "category". The input features to use are "paragraph" and "has_entity".
- 1b. The organization considers the results successful if the model performs better than a trivial baseline, avoids overfitting to the training dataset, and misclassifies no more than 10% of paragraphs into unrelated categories.
- 1c. They also want to determine which scalar performance metric would provide the most useful information to understand the algorithm's overall performance.

More specifically, for Task 1 (topic classification):
Model Building: Describe your approach to the topic classification task using machine learning techniques. Detail the final model hyperparameters, preferably in a table, and explain why you chose this specific algorithm. You may include hyperparameters from previous sections if relevant. Describe any experiments conducted to optimize the model, including hyperparameter optimization and comparisons with other models if applicable. Justify your design choices, including which hyperparameters to test, based on theory and/or experiments.

Model Evaluation: Evaluate the model's performance using a confusion matrix, a classification report, and other relevant metrics based on the dataset's characteristics and the client's specifications. Compare the results with a "trivial" baseline (e.g., random guess or majority class). Discuss how these metrics address the client's requirements and present the results in well-formatted figures and tables.

Conclusions:
- Determine if the model meets the client's definition of success (refer to point 1b from the task specifications).
- Recommend one scalar performance metric for the client to track the algorithm's performance.

2. Then, the client wants to explore **whether it would be possible to automatically detect if a given paragraph is written clearly enough**. They are planning to use the results to automatically reject edits and additions to the website’s knowledge base if they are not clear enough. However, they do not have any labels for this task outside of the first few rows of the dataset. So, they want you to **build a prototype by first labeling a subset of the data** (they give an optional suggestion of 100 data points), and then building a machine learning algorithm to predict these labels from the text and any other feature, as relevant. Specifically, they want you to use two labels: “`clear_enough`” and “`not_clear_enough`,” to denote the level of text clarity. You should add your labels in the column called “`text_clarity`.” This column will then be your output feature.
    1. They have heard there is now lots of interest about responsible use of machine learning. So, they would like you to review the ethical implications and risks of using an algorithm to automatically reject users’ work (for example, in terms of potential bias). Depending on the risks identified, they are open to consider applying the algorithm in a different way and are looking for suitable suggestions.
    2. The client will develop the prototype further if the algorithm produces results that do not overfit on the training data and are better than simply guessing the majority class all the time. They also want to know your top suggestion for improvement.
    3. The client is particularly interested in a prototype that includes some more advanced techniques (the main suggestions given are being able to make use of both labeled and unlabeled data points or using pre-trained word embeddings).

They want you to write the results of your analysis and implementation in a report. More details about what to include in the report are provided below.

Here is the structure of the dataset:
| FEATURE NAME       | BRIEF DESCRIPTION                                                                          |
|--------------------|---------------------------------------------------------------------------------------------|
| par_id             | Unique identifier for each paragraph to classify.                                           |
| paragraph          | Text to classify.                                                                           |
| has_entity         | Whether the text contains a reference to a product (yes/no), an organisation (yes/no), or a person (yes/no). |
| lexicon_count      | The number of words in the text.                                                            |
| difficult_words    | The number of difficult words in the text.                                                  |
| last_editor_gender | The gender of the latest person to edit the text.                                           |
| category           | The category into which the text should be classified.                                      |
| text_clarity       | The clarity level of the text. Very few data points are labelled at first.                 |