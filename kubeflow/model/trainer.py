from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Text
import absl
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx_bsl.tfxio import dataset_options
import keras
import keras.backend as K
from keras.utils import generic_utils
from PIL import Image
import PIL
from keras.models import Sequential, Model, load_model, model_from_json
from keras import layers
from keras.layers import Dense, Flatten, Dropout, Activation, LeakyReLU, Reshape, Concatenate, Input
from keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from keras import optimizers, losses
from copy import deepcopy
from tfx.components.trainer.fn_args_utils import FnArgs
from IPython.display import display, Image, Markdown, SVG
from keras.utils.vis_utils import model_to_dot
import itertools
import numpy as np
PATH_FOR_100PCT="gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/"
 
EETA_DEFAULT = 0.001     
MOMENTUM = 0.9            
# BATCH_SIZE = 64      
LEARNING_RATE = 0.1      
WEIGHT_DECAY = 0.0         
TOTAL_ITERATIONS = 20   
num_train_classes=4

# LARS Optimizer

class LARSOptimizer(tf.keras.optimizers.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
  """

  def __init__(self,
               learning_rate,
               momentum=MOMENTUM,
               use_nesterov=False,
               weight_decay=0.0,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               classic_momentum=True,
               eeta=EETA_DEFAULT,
               name="LARSOptimizer"):
    """Constructs a LARSOptimizer.

    Args:
      learning_rate: A `float` for learning rate.
      momentum: A `float` for momentum.
      use_nesterov: A 'Boolean' for whether to use nesterov momentum.
      weight_decay: A `float` for weight decay.
      exclude_from_weight_decay: A list of `string` for variable screening, if
          any of the string appears in a variable's name, the variable will be
          excluded for computing weight decay. For example, one could specify
          the list like ['batch_normalization', 'bias'] to exclude BN and bias
          from weight decay.
      exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
          for layer adaptation. If it is None, it will be defaulted the same as
          exclude_from_weight_decay.
      classic_momentum: A `boolean` for whether to use classic (or popular)
          momentum. The learning rate is applied during momeuntum update in
          classic momentum, but after momentum for popular momentum.
      eeta: A `float` for scaling of learning rate when computing trust ratio.
      name: The name for the scope.
    """
    super(LARSOptimizer, self).__init__(name)

    self._set_hyper("learning_rate", learning_rate)
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.use_nesterov = use_nesterov
    self.classic_momentum = classic_momentum
    self.eeta = eeta
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def _create_slots(self, var_list):
    for v in var_list:
      self.add_slot(v, "Momentum")

  def _resource_apply_dense(self, grad, param, apply_state=None):
    if grad is None or param is None:
      return tf.no_op()

    var_device, var_dtype = param.device, param.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))
    learning_rate = coefficients["lr_t"]

    param_name = param.name

    v = self.get_slot(param, "Momentum")

    if self._use_weight_decay(param_name):
      grad += self.weight_decay * param

    if self.classic_momentum:
      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        g_norm = tf.norm(grad, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(g_norm, 0), (self.eeta * w_norm / g_norm), 1.0),
            1.0)
      scaled_lr = learning_rate * trust_ratio

      next_v = tf.multiply(self.momentum, v) + scaled_lr * grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + scaled_lr * grad
      else:
        update = next_v
      next_param = param - update
    else:
      next_v = tf.multiply(self.momentum, v) + grad
      if self.use_nesterov:
        update = tf.multiply(self.momentum, next_v) + grad
      else:
        update = next_v

      trust_ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        v_norm = tf.norm(update, ord=2)
        trust_ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(v_norm, 0), (self.eeta * w_norm / v_norm), 1.0),
            1.0)
      scaled_lr = trust_ratio * learning_rate
      next_param = param - scaled_lr * update

    return tf.group(*[
        param.assign(next_param, use_locking=False),
        v.assign(next_v, use_locking=False)
    ])

  def _use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def get_config(self):
    config = super(LARSOptimizer, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "momentum": self.momentum,
        "classic_momentum": self.classic_momentum,
        "weight_decay": self.weight_decay,
        "eeta": self.eeta,
        "use_nesterov": self.use_nesterov,
    })
    return config

class Model(tf.keras.Model):
  def __init__(self, path):
    super(Model, self).__init__()
    self.saved_model = tf.saved_model.load(path)
    self.dense_layer = tf.keras.layers.Dense(units=num_train_classes, name="head_supervised_new")
    self.optimizer = LARSOptimizer(
      LEARNING_RATE,
      momentum=MOMENTUM,
      weight_decay=WEIGHT_DECAY,
      exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])

  def call(self, x):
    with tf.GradientTape() as tape:
      
      print(">>>>>>>>>>>>>>>>>>>",x[0])
      #outputs = self.saved_model(x['image'], trainable=False)
      outputs = self.saved_model(x[0], trainable=False)
      print(outputs)
      logits_t = self.dense_layer(outputs['final_avg_pool'])
      labels = tf.one_hot(x[1], num_train_classes)
      print("labels<<<<<<<",labels)
      print("logits",logits_t)
      loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(x[1], num_train_classes), logits=logits_t))
      dense_layer_weights = self.dense_layer.trainable_weights
      print('Variables to train:', dense_layer_weights)
      grads = tape.gradient(loss_t, dense_layer_weights)
      self.optimizer.apply_gradients(zip(grads, dense_layer_weights))
    return loss_t, x[0], logits_t, x[1]

model = Model(PATH_FOR_100PCT)

@tf.function
def train_step(x):
  return model(x)
#title TensorBoard Logger
class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
              tf.summary.scalar(tag, value,step=step)
              self.writer.flush()

#fine tune the linear layer
def finetune(train_dataset,validation_dataset,TOTAL_ITERATIONS=2,t_exp=""):
  y_true,y_pred = [], []

  # Train and Test Iterators
  train_iterator = iter(train_dataset)
  test_iterator = iter(validation_dataset)

  # Initialize TensorBoard Logger
  l = Logger("logs/scalars"+t_exp)

  # Training Loop
  for it in range(TOTAL_ITERATIONS):
    # Iterate over train images
    x = next(train_iterator)
    print("tamanna",x)
   
    train_loss, train_image, train_logits, train_labels = train_step(x)
    train_logits = train_logits.numpy()
    train_labels = train_labels.numpy()
    pred_train = train_logits.argmax(-1)
    num_train_correct = np.sum(pred_train == train_labels)
    y_true = np.append(y_true,train_labels)
    y_pred = np.append(y_pred,pred_train)
    total_train = train_labels.size
    l.log_scalar("Train Loss",train_loss,it+1)
    l.log_scalar("Train Top 1 Accuracy:",num_train_correct/float(total_train),it+1)
    print("[Iter {}] Loss: {} Top1 Accuracy: {}".format(it+1, train_loss, num_train_correct/float(total_train)))

  print("----------Evaluate the model----------") 
  for it in range(6): 
    # Iterate over test images
    x = next(test_iterator)
    test_loss, test_image, test_logits, test_labels = train_step(x)
    # print(test_loss, test_image, test_logits, test_labels)
    test_logits = test_logits.numpy()
    test_labels = test_labels.numpy()
    pred_test = test_logits.argmax(-1)
    num_test_correct = np.sum(pred_test == test_labels)
    total_test = test_labels.size
    l.log_scalar("Test Loss",test_loss,it+1)
    l.log_scalar("Test Top 1 Accuracy:",num_test_correct/float(total_test),it+1)
    print("[Iter {}] test_loss: {} test_top1_accuracy: {}".format(it+1, test_loss, num_test_correct/float(total_test)))
  
  print("----------Results----------")
  print(classification_report(y_true, y_pred, target_names=plant_patho_labels))
  # Plot the images and predictions
  fig, axes = plt.subplots(5, 1, figsize=(15, 15))
  for i in range(5):
    axes[i].imshow(train_image[i])
    true_text = plant_patho_labels[train_labels[i]]
    pred_text = plant_patho_labels[pred_train[i]]
    axes[i].axis('off')
    axes[i].text(256, 128, 'Truth: ' + true_text + '\n' + 'Pred: ' + pred_text)

# def feature_engg(features, label):
#   #Add new features
#   # feature = _preprocess(features)
#   image = tf.io.decode_raw(feature['image'], tf.uint8)
#   image=tf.cast(x=images, dtype=tf.float32)
#   image = tf.cast(image, tf.float32) * (1. / 255)
#   label = tf.cast(features['label'], tf.int32)

  # return(features, label)
# make_input function to be called for the trainer module
def make_input_fn(data_root, mode, vnum_epochs = None, batch_size=4):
    def decode_tfr(serialized_example):
      # define a parser
      features = tf.io.parse_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
        })
      image = tf.io.decode_raw(features['image'], tf.uint8)
      image = tf.reshape(image, [-1, 256, 256, 3])
      image=tf.cast(image, dtype=tf.float32)* (1. / 255)
      print("image.shape",image.shape)
      label = tf.cast(features['label'], tf.int32)
     


      return image,label
# input function for the trainer module file
    def _input_fn(v_test=False):
      # Get the list of files in this directory (all compressed TFRecord files)
      tfrecord_filenames = tf.io.gfile.glob(data_root)

      # Create a `TFRecordDataset` to read these files
      dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

      if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = vnum_epochs # indefinitely
      else:
        num_epochs = 1 # end-of-input after this

      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(buffer_size = batch_size)

      #Convert TFRecord data to dict
      dataset = dataset.map(decode_tfr)

      #Feature engineering
      # dataset = dataset.map(feature_engg)

      if mode == tf.estimator.ModeKeys.TRAIN:
          num_epochs = vnum_epochs # indefinitely
          dataset = dataset.shuffle(buffer_size = batch_size)
      else:
          num_epochs = 1 # end-of-input after this

      dataset = dataset.repeat(num_epochs)       
      
      #Begins - Uncomment for testing only -----------------------------------------------------<
      if v_test == True:
        print(next(dataset.__iter__()))
        
      #End - Uncomment for testing only -----------------------------------------------------<
      print("DATA SIZE")
      print(dataset.cardinality)
      return dataset
    return _input_fn
## function to save the model
def save_model(model, model_save_path):
  @tf.function
  def serving(image):
      ##Feature engineering

      payload = {
          'image': image
      }
      
      ## Prediction
      ##IF THERE IS AN ERROR IN NUMBER OF PARAMS PASSED HERE OR DATA TYPE THEN IT GIVES ERROR, "COULDN'T COMPUTE OUTPUT TENSOR"
      predictions = model(payload)
      return predictions

  serving = serving.get_concrete_function(image=tf.TensorSpec([None, 256,256,3], dtype=tf.uint8, name='image')
                                          )
# save the model on the path
  # version = "1"  #{'serving_default': call_output}
  tf.saved_model.save(
      model,
      model_save_path + "/",
      signatures=serving)
  
  
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.
  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  # print("Starting Training!!!!!!!!!!!!!!")
  # Getting custom arguments
  # batch_size = fn_args.custom_config['batch_size']
  # data_size = fn_args.custom_config['data_size']



#initializing the train dataset and the validation dataset
  train_dataset = make_input_fn(data_root = fn_args.train_files,
                      mode = tf.estimator.ModeKeys.TRAIN,
                      batch_size=4)()

  validation_dataset = make_input_fn(data_root = fn_args.eval_files,
                      mode = tf.estimator.ModeKeys.EVAL,
                      batch_size=4)()    

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():

    finetune(train_dataset,validation_dataset,TOTAL_ITERATIONS=5,t_exp="_100pct")

 
# Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')
 

  # show_images(train_dataset)
  
  



  save_model(generator,fn_args.serving_model_dir)
  print('serving model dir',fn_args.serving_model_dir)