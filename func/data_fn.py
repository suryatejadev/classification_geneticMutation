import pandas as pd
from modules import *

def load_data(data_path):
    # Load data
    train_variants = pd.read_csv(data_path+"training_variants")
    #test_variants  = pd.read_csv(data_path+"test_variants")
    #test_variants_2  = pd.read_csv(data_path+"test_variants_2.csv")
    train_text = pd.read_csv(data_path+"training_text", sep = "\|\|",
            engine = "python", skiprows = 1, names = ["ID", "Text"])
    #test_text = pd.read_csv(data_path+"test_text", sep = "\|\|",
    #        engine = "python", skiprows = 1, names = ["ID", "Text"])
    #test_text_2 = pd.read_csv(data_path+"test_text_2.csv", sep = "\|\|",
    #        engine = "python", skiprows = 1, names = ["ID", "Text"])
    # To find length of the text
    #train_text.loc[:, 'Text_count']  = train_text["Text"].apply(lambda x: len(x.split()))
    train_full = train_variants.merge(train_text, how="inner",
            left_on="ID", right_on="ID")
    #test_full = test_variants.merge(test_text, how="inner",
    #        left_on="ID", right_on="ID")
    #test_full_2 = test_variants_2.merge(test_text_2, how="inner",
    #        left_on="ID", right_on="ID")
    return train_full#test_full#,test_full_2

def filter_doc(doc):
   #tokenizer = RegexpTokenizer(r'\w+')
   tokenizer = RegexpTokenizer(r'[a-z]{3,}')
   doc_token = tokenizer.tokenize(doc.lower())
   return [w for w in doc_token if not w in stopwords.words('english')]

def load_w2v_data():
    path = 'outputs/word2vec_data/'
    corpus_train = []
    for i in range(3):
        t = time()
        pickle_in = open(path+'train_tx_'+str(i+1)+'.pickle','rb')
        corpus_train = corpus_train + pickle.load(pickle_in)
        print time()-t,' , ',len(corpus_train)
    corpus_test = []
    for i in range(3):
        t = time()
        pickle_in = open(path+'test_tx_'+str(i+1)+'.pickle','rb')
        corpus_test = corpus_test + pickle.load(pickle_in)
        print time()-t,' , ',len(corpus_test)
    return [corpus_train,corpus_test]

def load_d2v_data():
    path_train = 'outputs/doc2vec_data/'
    path_test = 'outputs/word2vec_data/'
    corpus_train = []
    for i in range(3):
        t = time()
        pickle_in = open(path_train+'train_tx_'+str(i+1)+'_d2v.pickle','rb')
        corpus_train = corpus_train + pickle.load(pickle_in)
        print time()-t,' , ',len(corpus_train)
    corpus_test = []
    for i in range(3):
        t = time()
        pickle_in = open(path_test+'test_tx_'+str(i+1)+'.pickle','rb')
        corpus_test = corpus_test + pickle.load(pickle_in)
        print time()-t,' , ',len(corpus_test)
    return [corpus_train,corpus_test]
