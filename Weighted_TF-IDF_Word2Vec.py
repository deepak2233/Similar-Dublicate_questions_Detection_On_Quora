questions = list(X_train['question1'].values.astype('U')) + list(X_train['question2'].values.astype('U'))
tfidf = TfidfVectorizer(lowercase=False)
tfidf.fit_transform(questions)
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

# Load the spacy model that you have installed
import en_core_web_sm
nlp = en_core_web_sm.load()

# each vector will be of length 94..
doc = nlp("This is some text that I am processing with Spacy")
#example
doc[3].vector.shape

X_train['question1'] = X_train['question1'].values.astype('str')
X_train['question2'] = X_train['question2'].values.astype('str')
X_test['question1'] = X_test['question1'].values.astype('str')
X_test['question2']  = X_test['question2'].values.astype('str')

nlp = spacy.load('en_core_web_sm')

vecs1 = []
# https://github.com/noamraph/tqdm
# tqdm is used to print the progress bar
for qu1 in tqdm(list(X_train['question1'])):
    doc1 = nlp(qu1)
    # 384 is the number of dimensions of vectors
    mean_vec1 = np.zeros([len(doc1), 96])
    for i,word1 in enumerate(doc1):
        # word2vec
        vec1 = word1.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        # compute final vec
        mean_vec1[i] += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)

X_train_glove_q1 = vecs1

nlp = spacy.load('en_core_web_sm')

vecs1 = []
# https://github.com/noamraph/tqdm
# tqdm is used to print the progress bar
for qu1 in tqdm(list(X_train['question2'])):
    doc1 = nlp(qu1)
    # 384 is the number of dimensions of vectors
    mean_vec1 = np.zeros([len(doc1), 96])
    for i,word1 in enumerate(doc1):
        # word2vec
        vec1 = word1.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        # compute final vec
        mean_vec1[i] += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)

X_train_glove_q2 = vecs1

nlp = spacy.load('en_core_web_sm')

vecs1 = []
# https://github.com/noamraph/tqdm
# tqdm is used to print the progress bar
for qu1 in tqdm(list(X_test['question1'])):
    doc1 = nlp(qu1)
    # 384 is the number of dimensions of vectors
    mean_vec1 = np.zeros([len(doc1), 96])
    for i,word1 in enumerate(doc1):
        # word2vec
        vec1 = word1.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        # compute final vec
        mean_vec1[i] += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)

X_test_glove_q1 = vecs1

nlp = spacy.load('en_core_web_sm')

vecs1 = []
# https://github.com/noamraph/tqdm
# tqdm is used to print the progress bar
for qu1 in tqdm(list(X_test['question2'])):
    doc1 = nlp(qu1)
    # 384 is the number of dimensions of vectors
    mean_vec1 = np.zeros([len(doc1), 96])
    for i,word1 in enumerate(doc1):
        # word2vec
        vec1 = word1.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        # compute final vec
        mean_vec1[i] += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)

X_test_glove_q2 = vecs1

X_train['q1_glove'] = X_train_glove_q1
X_train['q2_glove'] = X_train_glove_q2
X_test['q1_glove'] = X_test_glove_q1
X_test['q2_glove'] = X_test_glove_q2

train_glove = np.concatenate([np.array(X_train_glove_q1),np.array(X_train_glove_q2)],axis=1)
test_glove = np.concatenate([np.array(X_test_glove_q1),np.array(X_test_glove_q2)],axis=1)
train_glove.shape

glove_train_df = pd.DataFrame(train_glove,columns=[f'g_{i}' for i in range(train_glove.shape[1])])
glove_test_df = pd.DataFrame(test_glove,columns=[f'g_{i}' for i in range(test_glove.shape[1])])
glove_train_df.head()