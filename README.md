# NLP twitter Sentiment Analysis

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
  - **TF-IDF Embedding W/ Logistic Regression & Random Forest showed some promise so I conducted hyperparameter tuning which yeilded that Logistic Regresion in default setting has the highest validation accuracy of aproximatley 86%.**
  
- The previously defined **best classifer** is then applied on the entirety of the of US_Election_2020.csv dataset containing labeled 2020 US election relevant tweets and its **performance is sub optimal at 58%**. This is primarily due to computational contraints and dimensionality reduction requierments, the top 200 features from the generic tweets were used to train the model and only a randomly sampled eigth of the total dataset index was used, these features are not as informative when it comes to dictating sentiment in the US election tweets as they are insufficiently diverse and unable to effectivly explain the feature to sentiment mappings in the election relevant tweets.

- A Multi-Class Classification model is then created using the same steps as above to try and learn feature to negative sentiment reason mappings on the 2020 US election relevant tweets and its. The highest accuracy random forrest classification mod--el had an accuracy at 36% but overfit the data extremely. The logistic regression model had a similar accuracy with less overfitting characteristics but still at unreasonable levels.
  - The model did poorly in my opinion for the following reasons:
    - Unequal distribution of the labeled reasons with Covid significantly outnumbering the others
      - Scoring metric could have been changed to have a weighted accuracy however the class imbalance is too low to justify this. 
    - the sample size of the negative sentement labeled tweets with reasons is small and therefore models have a hard time generalizing on new data from the little they have learned from the small training set.

- Finnaly an MLP-3 is Built using Keras and TF in an attempt to build an even more compatant classifier however the validation accuracy is only 1% higher so the idea is scrapped. 

This was my first portfolio worthy project within the realm of NLP, model performance could be improved in the following ways:

- getting access to massively parallel processing (MPP) to speed things up and allow me to use the whole generic tweet set and more features as opposed to randomly sampling 1/8th of the overall index and only taking the top 200 most frequent features. (Can try DataBricks Pyspark)
- Using techniques such as word2vec or Glovo word embeddings to allow the model to better put sequence of words into context and improve prediction.
  - This will be the goal in my next NLP Project.

If you face issues viewing the notebook please follow the steps listed here
https://github.com/iurisegtovich/PyTherm-applied-thermodynamics/issues/11


