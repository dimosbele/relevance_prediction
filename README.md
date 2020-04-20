# relevance_prediction
Prediction of the relevance of a result with respect to a query

<b>Description</b> <br>
The task that was
to predict the relevance of a result with respect to a query. Each query is represented by a search term or
terms and the result is a specific product. Based on the relevance between the query and the answer, a
relevance score is assigned. The higher the score the better the relevance of the answer. The relevance is a
real number in the range [1,3], where 1 denotes minimum relevance and 3 is the maximum relevance. The task was to to predict the relevance score for an unknown combination between a query and a result.
The evaluation measure used is the Root-Mean-Square Error (RMSE).

<b>Datasets</b> <br>
There are three datasets available. 
1) The first one, train.csv, contains pairs of queries and answers and also contains the relevance score.
2) The second dataset is product_descriptions.csv, and for each different product it contains a textual description of the product.
3) The third file, attributes.csv, contains additional attributes for some of the products (some products may not contain additional attributes).

Thanks to Dimitris Orfanoudakis and Victor Pantzalis for the help in the project

Dimosthenis Beleveslis<br>
dimbele4@gmail.com
