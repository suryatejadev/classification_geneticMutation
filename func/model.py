from modules import *
from util_fn import *
from model_class import *

'''
Bag of words Algorithm
'''
def bag_of_words(data,parameters,model_cache):
    model_bow = TextEmbed_BagOfWords(parameters,model_cache)
    model_bow.build_vocab(data)
    print 'length of vocabulary = ',len(model_bow.model.get_feature_names())
    return model_bow

'''
Word2Vec Algorithm
'''
def word2vec(data,parameters,path,model_cache):
    # Initialize the model
    model_w2v = TextEmbed_Word2Vec(parameters,path,model_cache)
    load_data_cache = False
    if model_cache is None:
        # Preprocess the data
        #if load_data_cache==True:
        #    corpus_Tx = load_w2v_data()
        #else:
        corpus_Tx = model_w2v.preprocess_corpus(data,word_filter=True)
        print 'preproc done..',len(corpus_Tx[0]),len(corpus_Tx[1])
        # Save the preprocessed data
        pickle_path = open(path+'/cache/train_Tx.pickle','wb')
        pickle.dump(corpus_Tx[0], pickle_path)
        pickle_path = open(path+'/cache/test_Tx.pickle','wb')
        pickle.dump(corpus_Tx[1], pickle_path)
        print 'preproc data saved..'
        # Combine train and test data for training
        corpus_Tx = corpus_Tx[0] + corpus_Tx[1]
        print 'combine datas..',len(corpus_Tx)
        # Train the model
        model_w2v.train_Word2Vec(corpus_Tx)
    return model_w2v

'''
Doc2Vec Algorithm
'''
def doc2vec(data,parameters,path,model_cache):
    # Initialize the model
    model_d2v = TextEmbed_Doc2Vec(parameters,path,model_cache)
    load_data_cache = False
    if model_cache is None:
        # Preprocess the data
        if load_data_cache==True:
            corpus_Tx = load_d2v_data()
        else:
            corpus_Tx = model_d2v.preprocess_corpus(data,word_filter=True)
            print 'preproc done..',len(corpus_Tx[0]),len(corpus_Tx[1])
            # Save the preprocessed data
            pickle_path = open(path+'/cache/train_Tx.pickle','wb')
            pickle.dump(corpus_Tx[0], pickle_path)
            pickle_path = open(path+'/cache/test_Tx.pickle','wb')
            pickle.dump(corpus_Tx[1], pickle_path)
            print 'preproc data saved..'
        # Combine train and test data for training
        corpus_Tx = corpus_Tx[0] + corpus_Tx[1]
        print 'combine datas..',len(corpus_Tx)
        # Train the model
        model_d2v.train_Doc2Vec(corpus_Tx)
    return [model_d2v,corpus_Tx]

def train_model(data,model_type,parameters,path):
    # Create a directory to store the data of the model
    folder_path = 'outputs/'+path
    if os.path.exists(folder_path):
        t=time()
        model_cache = pickle.load(open(folder_path+'/cache/model.pickle','rb'))
        'model loaded in ',time()-t,'s'
    else:
        os.mkdir(folder_path)
        os.mkdir(folder_path+'/cache')
        model_cache = None

    # Build the model
    if model_type=='bag_of_words':
        model = bag_of_words(data,parameters,model_cache)
    elif model_type=='word2vec':
        model = word2vec(data,parameters,folder_path,model_cache)
    elif model_type=='doc2vec':
        model = doc2vec(data,parameters,folder_path,model_cache)
    else:
        model = None
        print 'model not present..'
    # Save the model only if it's trained, and not loaded
    if model_cache is None:
        pickle_path = open(folder_path+'/cache/model.pickle','wb')
        pickle.dump(model[0], pickle_path)
    return model



