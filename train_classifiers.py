from func.modules import *
from func.classifiers import *
from func.plot_confmatrix import *
from func.dataexp import *

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

    emb_list = ['count','tfidf','w2v','d2v','d2v_1']#[3]
    clf_name_list = ['ann','random_forest','svm','xgboost']#[4]
    gene_encoding_list = [True]#,False]#[1]

    output_full = np.zeros((len(emb_list),len(clf_name_list),len(gene_encoding_list),2))
    output = np.zeros((len(clf_name_list),len(gene_encoding_list),2))
    label0 = pickle.load(open('features/labels.pickle','rb'))
    gene0 = pickle.load(open('features/gene_enc.pickle','rb'))
    for iter_emb in range(len(emb_list)):
        emb = emb_list[iter_emb]
        data0 = pickle.load(open('features/feat_'+emb+'.pickle','rb'))
        for iter_clf_name in range(len(clf_name_list)):
            clf_name = clf_name_list[iter_clf_name]
            for iter_gene_encoding in range(len(gene_encoding_list)):
                gene_encoding = gene_encoding_list[iter_gene_encoding]
                print 'embedding = ',emb,', classifier = ',clf_name,', gene encoding = ',gene_encoding
                req_class = None#[0,1,3,6]
                #data1,label1 = select_class(data0,label0,gene0,req_class)

                if gene_encoding == True:
                        data = np.concatenate((gene0,data0),axis=1)
                if clf_name=='ann':
                    label,preds,probas=ann(data,label0)
                else:
                    clf = classifier(clf_name)
                    if clf is None:
                        clf = LogisticRegression()
                    label,preds,probas = get_probas_clf(clf,data,label0)
                plot_confmatrix(label, preds,title=' ',normalize=True)
                plt.savefig('output/'+emb+'/'+clf_name+'_dataexp1.jpg')
                output[iter_clf_name,iter_gene_encoding,:] = [log_loss(label, probas),accuracy_score(label, preds)]
                output_full[iter_emb,iter_clf_name,iter_gene_encoding,:] = [log_loss(label, probas),accuracy_score(label, preds)]
                print log_loss(label, probas),accuracy_score(label, preds)
                pickle.dump(output,open('output/'+emb+'/results_dataexp1.pickle','wb'))
    pickle.dump(output_full,open('output/results_full_dataexp1.pickle','wb'))

