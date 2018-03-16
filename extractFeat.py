from func.modules import *
from func.classifiers import *
from func.plot_confmatrix import *
from func.dataexp import *
from func.model import *

def select_class(data,label,gene,req_class=None):
    if req_class==None:
        data1 = np.concatenate((gene,data),axis=1)
        #data1,label1 = dataexp(data1,label)
        return data1,label
    (n_train,n_dim) = data.shape
    for i in range(len(req_class)):
        loc = np.where(label==req_class[i])[0]
        if i==0:
            x = data[loc]
            y = np.zeros((loc.size))
            z = gene[loc]
        else:
            x = np.concatenate((x,data[loc]),axis=0)
            y = np.concatenate((y,np.zeros((loc.size))+i),axis=0)
            z = np.concatenate((z,gene[loc]),axis=0)
    return x,y,z

def get_partition(N,req_class,class_len,class_train):
    labels = pickle.load(open('labels.pickle','rb'))
    index_train =[] #np.zeros((n_train*len(req_class)))
    index_val =[] #np.zeros((n_test*len(req_class)))
    print req_class
    for i in range(len(req_class)):
	loc_class = np.where(labels==req_class[i])[0]
    	print len(loc_class)
	n_class = np.round(class_len*len(loc_class)).astype(int)
    	n_train = np.round(class_train*n_class).astype(int)
	loc_class = loc_class[:n_class]
    	index_train.append(loc_class[:n_train])
    	index_val.append(loc_class[n_train:])
        print n_class,n_train,len(loc_class[n_train:])
    index_train = np.array([item for index_class in index_train for item in index_class])
    index_val = np.array([item for index_class in index_val for item in index_class])
    partition = {}
    partition['train'] = index_train
    partition['validation'] = index_val
    print len(partition['train']),len(partition['validation'])
    return partition

if __name__=='__main__':

    emb_list = ['count','tfidf','w2v','d2v','d2v_1']
    word_embeddings = {}
    word_embeddings['bag_of_words'] = {'feat_length':100, 'method':'tfidf', 'word_filter':True}
    word_embeddings['word2vec'] = {'size':100,'window':5,'min_count':2}
    word_embeddings['doc2vec'] = {'size':100,'window':5,'min_count':2}
    gene_encoding_list = [True,False]

    output_full = np.zeros((len(emb_list),len(clf_name_list),len(gene_encoding_list),2))
    output = np.zeros((len(clf_name_list),len(gene_encoding_list),2))
    label0 = pickle.load(open('features/labels.pickle','rb'))
    gene0 = pickle.load(open('features/gene_enc.pickle','rb'))

    embedding = 'word2vec'
    # Train the model
    model = train_model(data,embedding,word_embeddings[embedding],embedding)
    # Extract the features
    feat = model.feat_extract(data)
    # Save the features for classification
    pickle.dump(feat,open('features/feat_'+emb+'.pickle','wb'))



