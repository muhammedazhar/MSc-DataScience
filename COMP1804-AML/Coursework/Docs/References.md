Topic classification is a ‘supervised’ machine learning technique, one that needs training before being able to automatically analyze texts.
A topic classification model could also be used to determine what customers are talking about in customer reviews, open-ended survey responses, and on social media, to name just a few.
This classify text paragraphs into specific topics based on the content and any mentioned entities such as persons, organizations, or products.
The model should consider not just the textual content but also whether the paragraph mentions specific entities, enhancing its prediction accuracy.
OpenAI. (2024). ChatGPT (4) [Large language model]. https://chat.openai.com
https://stackoverflow.com/questions/19073683/how-to-fix-overlapping-annotations-text
https://adjusttext.readthedocs.io/en/latest/
The `has_entity` column contains unique values representing different combinations of entities (organization, product, and person) present or absent in paragraphs. The goal is to determine the total number of possible combinations based on these values and understand their significance.
Identify Entities: The entities considered are organization (ORG), product (PRODUCT), and person (PERSON).
Possible States: Each entity can either be present (YES) or absent (NO), resulting in 2 possible states for each entity.
Total combinations = 2^3 = 8
Total combinations = 2^{3} = 8

Evaluation: https://youtu.be/LbX4X71-TFI?si=rcFn0sXIei3-FvCJ