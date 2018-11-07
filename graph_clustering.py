
# coding: utf-8

# In[2]:


import numpy as np
import scipy.sparse as sp
import tensorflow as tf


# In[3]:


path="./data/cora/"
dataset="cora"


# In[4]:


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


# In[5]:


idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                    dtype=np.dtype(str))
labels = encode_onehot(idx_features_labels[:, -1])


# In[6]:


idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}


# In[7]:


edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
edges_transpose = np.transpose(edges)


# In[8]:


A = np.zeros(shape=(idx.shape[0],idx.shape[0]))
for e in edges:
    A[e[0]][e[1]]=1
X = idx_features_labels[:, 1:-1]
Y=labels


# In[9]:


D=np.eye(idx.shape[0])
I=np.eye(idx.shape[0])
for e in edges:
    D[e[0]][e[0]]=np.sum(edges_transpose[0] == e[0])
AI = A + I
D_inv = np.linalg.inv(D)
D_half = np.dot(D**0.5,D_inv)
A_t = np.dot(np.dot(D_half,AI),D_half)


# In[10]:


# Deg[i] means the no of edges terminating on that vertex ie np.sum(edges_transpose[0]==i)
# D is diag(Deg[i]) $\forall$ i
#AI = A + I .. defining that a node i is connected to itself


# In[11]:


# X is N X D , Y is N X K , A_t is A X N .. good to go, we need degree matrix Deg


# In[12]:


print A_t.shape , X.shape


# In[13]:


F = 50 # the dimensions of latent variable
H = 500 # the dimension for Hidden unit
D = X.shape[1]
# W0 would be D X H, W1 would be H X F


# In[14]:


w0 = tf.Variable(tf.random_uniform([D, H]),trainable=True)
w1 = tf.Variable(tf.random_uniform([H, F]),trainable=True)
A_t_tf = tf.convert_to_tensor(A_t, dtype=tf.float32)
X_tf = tf.convert_to_tensor(X,dtype=tf.float32)


# In[15]:


# First layer :
z1 = tf.matmul(tf.matmul(A_t_tf,X_tf),w0)
a1 = tf.nn.relu(z1)


# In[16]:


#Second Layer :
z2 = tf.matmul(tf.matmul(A_t_tf,a1),w1)
a2 = tf.nn.softmax(z2)

#Final assignent to Z, it has to be N X F
Z = a2


# In[24]:


#defining the loss function :
mat = tf.matmul(Z,tf.transpose(Z))
mat_flat=tf.reshape(mat, [-1])
A_t_tf_flat = tf.reshape(A_t_tf,[-1])
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=A_t_tf_flat,logits=mat_flat))


# In[25]:


opt = tf.train.AdamOptimizer(0.00001).minimize(loss)


# In[26]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        print(sess.run([loss]))
        sess.run(opt)


# In[43]:


# x = tf.Variable(10.0, trainable=True)
# y = tf.Variable(10.0, trainable=True)


# fx = tf.nn.l2_loss(tf.exp(x)+tf.multiply(x,y)-20)
# loss = fx
# opt = tf.train.GradientDescentOptimizer(0.000002).minimize(fx)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(200):
#         print(sess.run([x,y,loss]))
#         sess.run(opt)

