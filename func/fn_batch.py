import numpy as np
from keras.utils import np_utils
import pickle

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 1000, dim_y = 100, batch_size = 32, nb_classes = 9, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.nb_classes = nb_classes
      self.gene_data = pickle.load(open('features/gene_enc.pickle','rb'))

  def generate(self, labels, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(labels, list_IDs_temp)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def load_feat(self,file_name):
      f = open(file_name,'rb')
      feat = []
      for line in f:
          a = line.replace('[','').replace(']','').split()
          for k in a:
              feat.append(float(k))
      feat = np.array(feat)
      return feat.reshape(int(len(feat)/100),100)

  def zeropad(self,a,n):
      (nr,nc) = a.shape
      if nr<n:
          a_pad = np.zeros((n,self.dim_y-2))
          a_pad[:nr,:] = a
          return a_pad
      else:
          return a[:n,:]  

  def __data_generation(self, labels, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y))
      y = np.empty((self.batch_size), dtype = int)
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
          a= self.load_feat('../outputs/w2v_'+str(int(ID+1))+'.csv')
          a = self.zeropad(a,self.dim_x)
          X[i, :, 2:] = a
          X[i,:,0] = self.gene_data[int(ID),0]
          X[i,:,1] = self.gene_data[int(ID),1]
          # Store class
          y[i] = labels[int(ID)]
      y = np_utils.to_categorical(y, self.nb_classes)
      return X, y


