# Predictive-Modeling-for-Customer-Engagement-with-Offers-using-AMEX-data
Built a personalized offer ranking system to predict the probability of a customer clicking on an Amex digital offer. Combined customer profiles, transaction history, and offer features to train a ranking model evaluated using MAP@7 for top offer relevance.

💳 Amex Offer Click Prediction and Ranking (MAP@7)
This project aims to build a personalized recommendation system that predicts the likelihood of a customer clicking on digital offers provided by American Express (Amex). The model helps rank the most relevant offers for each customer, enhancing engagement and conversion. Final model performance is evaluated using Mean Average Precision at 7 (MAP@7).

📌 Problem Statement
Given customer characteristics, offer attributes, and transaction history, the task is to:

Predict the probability of clicking on an offer that a customer has seen.

Rank the offers so that top-ranked ones are the most likely to be clicked.

Improve the overall engagement by showing personalized, high-relevance offers.

🔄 Workflow Overview
1. Data Sources
Three datasets were provided:

Customer Information: Demographics, income, location, etc.

Transaction History: Merchant ID, transaction value, product category, timestamps, etc.

Offer Metadata: Offer category, start/end date, reward type, channels, etc.

2. Feature Engineering
Aggregated customer spending behavior (e.g., last 7/30 days)

Encoded customer-offer interactions (e.g., category match, recency)

Created temporal features (day of week, hour of impression)

Calculated customer historical click-through patterns

3. Modeling Approach
Ranking Objective: Trained a model using XGBoost (rank:pairwise) to generate ranked lists of offers per customer.

Alternative approaches: Binary classifiers with probability → ranked output

Used group-wise cross-validation to ensure customer-level separation

🧪 Evaluation Metric: MAP@7
The output is a ranked list of 7 offers per customer. Performance is measured using Mean Average Precision at 7:

MAP@7 gives higher weight to correctly predicted offers in top positions

Final score = average over all customers in test set
