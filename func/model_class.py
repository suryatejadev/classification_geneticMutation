from modules import *
from data_fn import *

'''
Bag of words Algorithm
1. Frequency
2. TF-IDF
'''
class TextEmbed_BagOfWords:
    def __init__(self,parameters,model_cache=None):
        self.model = None
        self.feat_length = parameters['feat_length']
        self.method = parameters['method']
        if model_cache is not None:
            self.model = model_cache.model
            self.feat_train = model_cache.feat_train
            self.feat_test  = model_cache.feat_test
        self.word_filter = parameters['word_filter']

    def build_vocab(self,data):
        # If model_cache==None, build the model
        # Else, the model can be loaded from 'cache/'
        if self.model==None:
            print 'Building new model..'
            t=time()
            data_concat = pd.Series(pd.concat(data).values)
            print 'concatenated data in ',time()-t,' seconds.'
            t=time()
            if self.word_filter==True:
                for i in range(len(data_concat)):
                    data_concat[i] = ' '.join(filter_doc(data_concat[i]))
            print 'filtered data in ',time()-t,' seconds.'
            if self.method=='count':
                self.model = CountVectorizer(analyzer="word",
                        tokenizer=nltk.word_tokenize, preprocessor=None,
                        max_features=None)
            if self.method=='tf_idf':
                self.model = TfidfVectorizer(analyzer="word",
                        tokenizer=nltk.word_tokenize, preprocessor=None,
                        max_features=None)
            t=time()
            feat = self.model.fit_transform(data_concat)
            print 'Fit data in ',time()-t,' seconds.'
            n_train = len(data[0])
            self.feat_train = feat[:n_train]
            self.feat_test = feat[n_train:]
        else:
            print 'Using existing model..'

    def feat_extract(self,data,path=None):
    	if path==None:
			# Perform dimensionaly reduction using truncated SVD
			svd = TruncatedSVD(n_components=self.feat_length,
			n_iter=25, random_state=12)
			if len(data)==self.feat_train.shape[0]:
				return svd.fit_transform(self.feat_train)
			return svd.fit_transform(self.feat_test)
    	else:
		    return pickle.load(open(path,'rb'))

'''
Word2Vec Algorithm
'''
class TextEmbed_Word2Vec:
    def __init__(self,parameters,path,model_cache=None):
        self.model = None
        self.size = parameters['size']
        self.window = parameters['window']
        self.min_count = parameters['min_count']
        self.path = path
        self.n_train=None
        self.n_test=None
        self.word_filter=True
        if model_cache is not None:
            self.model = model_cache.model
            self.n_train = model_cache.n_train
            self.n_test = model_cache.n_test

    def word_tokenize(self,data):
        index = data[0]
        doc = data[1]
        print index
        doc_Tx = []
        for sentence_token in nltk.sent_tokenize(doc.decode('utf-8')):
            if self.word_filter==True:
                word_token = filter_doc(sentence_token)
            else:
                word_token = nltk.word_tokenize(sentence_token)
            for word in word_token:
                doc_Tx.append(word)
        return doc_Tx

    def preprocess_corpus(self,corpus,word_filter):
        corpus_Tx_train = []
        self.word_filter = word_filter
        print 'train preproc..'
        p = Pool()
        corpus_Tx_train.append(p.map(self.word_tokenize,itertools.izip(range(len(corpus[0])),corpus[0].tolist())))
        corpus_Tx_train = corpus_Tx_train[0]
        self.n_train = len(corpus[0])

        corpus_Tx_test = []
        print 'test preproc..'
        p = Pool()
        corpus_Tx_test.append(p.map(self.word_tokenize,itertools.izip(range(len(corpus[1])),corpus[1].tolist())))
        corpus_Tx_test = corpus_Tx_test[0]
        self.n_test = len(corpus[1])
        return [corpus_Tx_train,corpus_Tx_test]

    def train_Word2Vec(self,corpus):
        # If there is no saved model, train one using the corpus
        if self.model==None:
            print('model not found. training model')
            self.model = gensim.models.Word2Vec(
                corpus, size=self.size, window=self.window,
                min_count=self.min_count, workers=4)
            print('Model done training. Saving to disk')
        else:
            print('Loading saved model..')

    def feat_extract(self,corpus):
        load_data_cache = False
        # vectorize data to feed to word2vec model
        data_Tx = self.preprocess_corpus(corpus,self.word_filter)

        if len(corpus)==self.n_train:
            data = data_Tx[0]
        else:
            data = data_Tx[1]

        '''if load_data_cache==True:
            if len(corpus)==self.n_train:
                data = load_w2v_data()[0]
            else:
                data = load_w2v_data()[1]
        else:
            if len(corpus)==self.n_train:
                path = self.path+'/cache/train_Tx.pickle'
            else:
                path = self.path+'/cache/test_Tx.pickle'
            data = pickle.load(open(path,'rb'))'''
        # List of features of all documents
        feat_doc = []
        # Iterate over documents in the corpus
        for doc in corpus:
            # List of features of all words in a document
            feat_word = []
            # Iterate over words in a document
            for word in doc:
                # if the word is in the vocab, append its feature
                if word in self.model.wv:
                    feat_word.append(self.model.wv[word])
                # else, append an all-zero feature
                else:
                    feat_word.append(np.zeros(self.size))
            # feature of doc = mean(features of words in the doc)
            feat_doc.append(np.mean(feat_word,axis=0))
        return np.array(feat_doc)

'''
Doc2Vec Algorithm
'''
class TextEmbed_Doc2Vec:
    def __init__(self,parameters,path,model_cache=None):
        self.model = None
        self.size = parameters['size']
        self.window = parameters['window']
        self.min_count = parameters['min_count']
        self.path = path
        self.n_train=None
        self.n_test=None
        self.iter = parameters['iter']
        self.word_filter = True
        self.set_d2v_flag = False

        if model_cache is not None:
            self.model = model_cache.model
            self.n_train = model_cache.n_train
            self.n_test = model_cache.n_test

    def word_tokenize_d2v(self,data):
        index = data[0]
        doc = data[1]
        print index
        doc_Tx = []
        for sentence_token in nltk.sent_tokenize(doc.decode('utf-8')):
            if self.word_filter==True:
                word_token = filter_doc(sentence_token)
            else:
                word_token = nltk.word_tokenize(sentence_token)
            for word in word_token:
                doc_Tx.append(word)
        if self.set_d2v_flag==True:
            return gensim.models.doc2vec.TaggedDocument(doc_Tx,[index])
        else:
            return doc_Tx

    def preprocess_corpus(self,corpus,word_filter):
        self.n_train = len(corpus[0])
        self.n_test = len(corpus[1])

        corpus_Tx_train = []
        self.word_filter = word_filter
        print 'train preproc..'
        self.set_d2v_flag=True
        p = Pool()
        train_labels = range(self.n_train)
        corpus_Tx_train.append(p.map(self.word_tokenize_d2v,itertools.izip(train_labels,corpus[0].tolist())))
        corpus_Tx_train = corpus_Tx_train[0]
        print len(corpus_Tx_train),corpus_Tx_train

        corpus_Tx_test = []
        print 'test preproc..'
        #self.set_d2v_flag = False
        p = Pool()
        test_labels = range(self.n_train,self.n_train+self.n_test)
        corpus_Tx_test.append(p.map(self.word_tokenize_d2v,itertools.izip(test_labels,corpus[1].tolist())))
        corpus_Tx_test = corpus_Tx_test[0]
        print len(corpus_Tx_test),corpus_Tx_test

        return [corpus_Tx_train,corpus_Tx_test]

    def train_Doc2Vec(self,corpus):
        # If there is no saved model, train one using the corpus
        if self.model==None:
            print('model not found. training model')
            self.model = gensim.models.doc2vec.Doc2Vec(
                size=self.size, min_count=self.min_count,
                window=self.window,iter=self.iter)
            self.model.build_vocab(corpus)
            self.model.train(corpus, total_examples=self.model.corpus_count,
                             epochs=self.model.iter)
            print('Model done training. Saving to disk')
        else:
            print('Loading saved model..')

    def feat_extract(self,corpus):
        # List of features of all documents
        feat_doc = []
        # Iterate over documents in the corpus
        for doc in corpus:
            doc = doc.words
            # List of features of all words in a document
            feat_word = []
            # Iterate over words in a document
            for word in doc:
                # if the word is in the vocab, append its feature
                if word in self.model.wv:
                    feat_word.append(self.model.wv[word])
                # else, append an all-zero feature
                else:
                    feat_word.append(np.zeros(self.size))
            # feature of doc = mean(features of words in the doc)
            feat_doc.append(np.mean(feat_word,axis=0))
        return np.array(feat_doc)

    # Doc2vec can also infer the embeddings given a document.
    # This method extracts features using inference for embedding extraction
    def feature_extract(self,corpus):
        # List of features of all documents
        feat_doc = []
        # Iterate over documents in the corpus
        for doc in range(len(corpus)):
            # Infer a feature vector for the document
            print len(corpus[doc][0])
            feat_doc.append(self.model.infer_vector(corpus[doc][0]))
        return np.array(feat_doc)

