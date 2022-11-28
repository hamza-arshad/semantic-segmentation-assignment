import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import load_img

class DataGenerator(Sequence):
  'Generates data for Keras'
  
  def __init__(self, pair, transformer = None, batch_size=16, dim=(192,256,3), shuffle=True):
    'Initialization'
    self.dim = dim
    self.pair = pair
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.transformer = transformer
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.pair) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [k for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)

    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.pair))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    batch_imgs = list()
    batch_labels = list()

    # Generate data
    for i in list_IDs_temp:
      # Store sample
      img = load_img(self.pair[i][0])
      label = load_img(self.pair[i][1], color_mode='grayscale')
      if self.transformer is not None:
        transformed = self.transformer(image=np.array(img), mask=np.array(label))
        img = transformed['image']
        label = transformed['mask']
      
      batch_imgs.append(img)
      label = to_categorical(label, num_classes=12)
      batch_labels.append(np.array(label))
        
    return np.array(batch_imgs) ,np.array(batch_labels)