# Importing the libraries
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt 
import tensorflow as tf
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
tf.__version__
tf.reset_default_graph()


# Defining the Mexican Hat Wavelet Function
def Mexican(z):
    return tf.multiply(tf.subtract(tf.constant(1.0), tf.pow(z,2)),
    			tf.exp(tf.multiply(tf.constant(-0.5), tf.pow(z,2))))


# Loading the Dataframe
data = pd.read_csv('data.csv')


# Separating the Target Value from the dataframe
y = demand = data['demand']
X = data.drop('demand', axis=1)


# Divinding the Training and test data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size = 0.3)



# Some terms
n_features = X_train.shape[1]
n_hidden = 30                
n_class = 1
lr = 0.01        #Learning rate



# Tensorflow Placeholders
x = tf.placeholder(tf.float32, [None, n_features], name = 'X_label')
y_true = tf.placeholder(tf.float32, [None, n_class], name = 'y_label')



# Function to create a new weight
def new_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=3),name = name)
    


# Defining the various weights
W_trans = new_weights([n_hidden, n_features], 'w_trans')
W_dilat = new_weights([n_hidden, n_features],'W_dilat')
W_lin = new_weights([n_class, n_hidden], 'w_lin')
W_direct = new_weights([n_class, n_features], 'w_dir')

biases = tf.Variable(tf.constant(0.05, shape=[1]))



# Activation of hidden layer
act_hidd = tf.reshape(tf.reduce_prod(Mexican(tf.divide(tf.subtract(x, W_trans), W_dilat)), axis = 1), shape=[n_hidden,1])

# Final output
logits = tf.transpose(tf.matmul(W_lin, act_hidd)) + biases + tf.matmul(x,tf.transpose(W_direct))

# Cost
cost = tf.reduce_mean(tf.square(logits-y_true))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)


# Running the Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

costy = []
def optimize(num_iterations):
    
    for j in range(num_iterations):
        print('Epoch: ', j)
        tot_cost = 0
        for x1,y1 in zip(X_train, y_train):
            feed_dict_train = {x:x1.reshape(1,9), y_true:y1.reshape(1,1)}
            _, my_cost = sess.run([optimizer, cost] ,feed_dict=feed_dict_train)
            tot_cost += my_cost/120
        print(tot_cost)

        costy.append(tot_cost)

optimize(num_iterations=10)
plt.plot(costy)
plt.show(block=True)