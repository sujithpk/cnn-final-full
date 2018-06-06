import tensorflow as tf
import numpy as np
import inp_file_test

num_test = 14 #no of test datasets
n_test = num_test*3 # no of data to be tested

# Network Parameters
num_input = 784 # accel data input (28*28)
num_classes = 12 # accel total classes , 3*3

#Getting inp data
data = inp_file_test.read_inp(n_test,num_classes,one_hot=False)

x_set = data.test._acdata
x_set = np.reshape(x_set,[n_test,28,28,1])

def largest(a):
	max = a[0]
	pos = 0
	for i in range(num_classes):
		if(a[i]>max):
			max = a[i]
			pos = i
	return pos

def test_fn(x_set):

	#The input is of shape [None, height,width, num_channels]
	x_batch = np.reshape(x_set,[1,28,28,1])
	# Restoring the saved model 
	sess = tf.Session()
	# Recreating the network graph
	saver = tf.train.import_meta_graph('cnn-model.meta')
	# Loading the weights
	saver.restore(sess, tf.train.latest_checkpoint('./'))

	# Accessing the restored default graph
	graph = tf.get_default_graph()

	# o/p tensor in original graph
	y_pred = graph.get_tensor_by_name("y_pred:0")

	# Feeding inputs to the input placeholders
	x= graph.get_tensor_by_name("x:0") 
	y_true = graph.get_tensor_by_name("y_true:0") 
	y_test_images = np.zeros((1, num_classes)) 

	# Creating feed_dict & running the session to get 'result'
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)

	return largest(result[0])

# Testing
preds=np.zeros(n_test)  
for i in range(n_test):
    curr_pred = test_fn (x_set[i])   
    preds[i]=curr_pred  #preds: 0 to 12

#assumption: only accel values are used (x,y,z)
preds_edit = np.zeros(n_test)
#convert to real test labels
for i in range(n_test):
    for j in range(4):
        if preds[i]==3*j+0 or preds[i]==3*j+1 or preds[i]==3*j+2:
            preds_edit[i] = j    #preds_edit: 0 to 3

preds_final = np.zeros(num_test)
cnt=0
for i in range(0,n_test,3):
    avg = 0.0
    avg= (preds_edit[i] + preds_edit[i+1] + preds_edit[i+2] )/3
    avg = int(round(avg))
    preds_final[cnt] = avg  #preds_final 0 to 3
    cnt=cnt+1

#Display the predicted classes
for i in range(num_test):
    print('Test data:',i+1, "   Predicted Speed:", int(preds_final[i]))

#Finding accuracy
test_labels= np.array([0,1,2,3,1,3,1,2,2,1,3,1,2,0])
adn=0.0
for i in range(num_test):
    if preds_final[i]==test_labels[i] :
        adn=adn+1.0

print('\nAccuracy of testing :',adn/num_test*100,'%') 
