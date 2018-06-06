#update num_train

from __future__ import division, print_function, absolute_import
import inp_file_train
import tensorflow as tf
import numpy as np

num_train =29 #no of training datasets in for each speed
n_train = num_train*4*3 # 261 - no of datasets in training process

# Training Parameters
batch_size = 21 # or 29

# Network Parameters
num_input = 784 # accel data input (28*28)
num_classes = 12 # accel total classes , 3*3

#Getting inp data
data = inp_file_train.read_inp(n_train,num_classes,one_hot=False)

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, 28,28,1], name='x')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

num_channels=1

##Network graph params
filter_size_conv1 = 5 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 64

fc_layer_size = 1024

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):     
   
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    # Create conv layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
    layer += biases

    # max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    # Output of pooling is fed to Relu, the activation function
    layer = tf.nn.relu(layer)

    return layer

def create_flatten_layer(layer):
    # getting shape from the previous layer.
    layer_shape = layer.get_shape()

    # No. of features = height * width* channels
    num_features = layer_shape[1:4].num_elements()

    # Flatten the layer
    layer = tf.reshape(layer, [-1, num_features])

    return layer

def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # For FC layer i/p=x and o/p=wx+b
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)
          
layer_flat = create_flatten_layer(layer_conv2)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)

session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer()) 

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


total_iterations = 0
saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch = data.validation.next_batch(batch_size)

        # Reshape
        x_batch1=np.reshape(x_batch, [-1,28,28,1])
        x_valid_batch1=np.reshape(x_valid_batch, [-1,28,28,1])
        
        feed_dict_tr = {x: x_batch1,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch1,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train._num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train._num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './cnn-model') 


    total_iterations += num_iteration

train(num_iteration=400)

print('\n..Training completed..\n')
