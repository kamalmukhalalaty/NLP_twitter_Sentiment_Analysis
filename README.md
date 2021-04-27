# NLP_twitter_Sentiment_Analysis

In this repository I showcase a notebook where I built a NLP tweet sentiment analysis classifier. 

- The binary classifier is trained and validated on generic tweets from the file named sentiment_analysis.csv. 
  - The extracted corpus is modeled via both Bag of words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF) embeddings.
  - top 200 features were selected based on frequency and the following models were applied using scikit's default settings:
    - LogisticRegression()
    - RandomForestClassifier()
    - DecisionTree
    - SVM
    - KNN
    - XGBoost W/Logistic objective
  -TF-IDF Embedding W/ Logistic Regression & Random Forest showed some promise so I conducted hyperparameter tuning which yeilded that Logistic Regresion in default setting has the highest validation accuracy of aproximatley 86%.
  
- The previously defined best classifer is then applied on the entirety of the of US_Election_2020.csv dataset containing labeled 2020 US election relevant tweets and its performance is not as good at 58%. This is primarily due to computational contraints and dimensionality reduction requierments, the top 200 features from the generic tweets were used to train the model, these features are not as informative when in comes to dictating sentiment in the US election tweets as they are insufficiently diverse and unable to effectivly explain the relativly specific feature to sentiment mappings in the election relevant tweets.

- A Multi-Class Classification model is created using the same steps as above to try and learn feature to negative sentiment reason mappings.

- Finnaly an MLP-3 is Built using Keras and TF in an attempt to build an even more compatant classifier however the va;idation accuracy is only 1% higher so the idea is scrapped. 

This was my first portfolio worthy project within the realm of NLP, model performance could be improved in the following ways:

- getting access to massively parallel processing (MPP) to speed things up and allow me to use the whole generic tweet set and more features as opposed to randomly sampling 1/8th of the overall index and only taking the top 200 most frequent features. (Can try DataBricks Pyspark)
- Using techniques such as word2vec or Glovo word embeddings to allow the model to better put sequence of words into context and improve prediction.
  - This will be the goal in my next NLP Project.


