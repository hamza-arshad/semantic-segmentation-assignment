import numpy as np
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import tf.keras.backend as K
from tf.keras.preprocessing.image import load_img, img_to_array
from random import choice

def get_image_with_mask_imposed(img, target, classes, colors):
  for ll in range(classes):
    mask=(target==ll)
    img=mark_boundaries(img, mask, color=colors[ll], mode='thick')
  return img

def f1score(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
  return f1_val

def dice_coef(y_true, y_pred):
  y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
  y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
  intersection = tf.reduce_sum(y_true_f * y_pred_f)
  dice_coef = (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)
  return dice_coef

def specifity(y_true, y_pred):
  neg_y_true = 1 - y_true
  neg_y_pred = 1 - y_pred
  fp = K.sum(neg_y_true * y_pred)
  tn = K.sum(neg_y_true * neg_y_pred)
  spec = tn / (tn + fp + K.epsilon())
  return spec

def sensitivity(y_true, y_pred):
  neg_y_pred = 1 - y_pred
  tp = K.sum(y_true * y_pred)
  fn = K.sum(neg_y_pred * y_true)
  sens = tp / (tp + fn + K.epsilon())
  return sens

def make_pair(img,label):
  pairs = []
  for i in range(len(img)):
    pairs.append((img[i], label[i]))
  
  return pairs

def loadAndProcessImages(pair, img_size):
  temp = choice(pair)
  img = img_to_array(load_img(temp[0], target_size=img_size))
  mask = img_to_array(load_img(temp[1], target_size=img_size, color_mode='grayscale'))

  return img, mask

def make_prediction(model,img_path,shape):
    img= img_to_array(load_img(img_path , target_size= shape))/255.
    img = np.expand_dims(img,axis=0)
    labels = model.predict(img)
    labels = np.argmax(labels[0],axis=2)
    return labels