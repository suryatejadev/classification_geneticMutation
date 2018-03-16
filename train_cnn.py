import numpy as np
import cPickle as pickle
from keras.models import Sequential
from fn_batch import DataGenerator
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.utils import np_utils
from func.fn_batch import *
from func.func import *

def remove_index(x):
	rem_index = pickle.load(open('remove_index.pickle','rb'))
	new_x = []
	for i in range(len(x)):
		if x[i] not in rem_index:
			new_x.append(x[i])
	return np.array(new_x)

# [568,452,89,686,242,275,953,19,37]
def get_labels(req_class):
	labels = pickle.load(open('labels.pickle','rb'))
	for i in range(len(req_class)):
		loc = np.where(labels==req_class[i])[0]
		labels[loc] = i
	return labels

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

	index_train = remove_index(index_train)
	index_val = remove_index(index_val)

	#for i in range(len(req_class)):
	#	loc_class = np.where(labels==req_class[i])[0]
	#	print len(loc_class)
	#	index_train[i*n_train:(i+1)*n_train] = loc_class[:n_train]
	#	index_test[i*n_test:(i+1)*n_test] = loc_class[n_train:n_train+n_test]
	partition = {}
	partition['train'] = index_train
	partition['validation'] = index_val
	print len(partition['train']),len(partition['validation'])
	return partition

# set parameters:
batch_size = 128
maxlen = 1000
embedding_dim = 102
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5
N = 3321
#req_class = [0,1,2,3,4,5,6,7,8]
req_class = [0,1,3,6]
nb_classes = len(req_class)
class_len = 1
class_train = 0.7
params = {'dim_x': maxlen,
          'dim_y': embedding_dim,
          'batch_size': batch_size,'nb_classes':nb_classes,
          'shuffle': True}

# Datasets
partition = get_partition(N,req_class,class_len,class_train)# IDs
labels = get_labels(req_class)# Labels

# Generators
training_generator = DataGenerator(**params).generate(labels, partition['train'])
validation_generator = DataGenerator(**params).generate(labels, partition['validation'])

score_train = []
acc_train = []
score_test = []
acc_test = []

print('Build model...')
model = Sequential()
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,input_shape=(maxlen,embedding_dim)))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model on dataset
out,wt = model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(partition['train'])//batch_size,epochs=epochs,
                    validation_data = validation_generator,
                    validation_steps = len(partition['validation'])//batch_size,verbose = 1)

model.save('model_cnn_5')
print out.history
pickle.dump(wt,open('wts_cnn_5.pickle','wb'))

