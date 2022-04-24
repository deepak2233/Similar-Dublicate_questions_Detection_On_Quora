# Similar-Dublicate_questions_Detection_On_Quora


# **Problem Statement**
Identify which questions asked on Quora are duplicates of questions that have already been asked.This could be useful to instantly provide answers to questions that have already been answered.We are tasked with predicting whether a pair of questions are duplicates or not.

---

# **Prerocessing** 

* Regular Expression 
* Removing html tags
* Removing Punctuations
* Performing stemming
* Removing Stopwords
* Expanding contractions etc.
---

# **Methodlgy**

* Used Hand-craft Featurs and Fuzzy String matching based features 
* TF-IDF word Embedding and Weighted TF-IDF Glove(Word2vec) vectorization
* ML Model -> SVM and Logistic Regression with Hyparparameter Tuning.

---

**Results**
* Performance by using TF-IDF
* Performance by Using TF-IDF weighted Glove(W2Vec)
