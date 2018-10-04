#-*-coding:utf-8-*-
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# print(type(mnist))
# print(mnist)

#神经网络训练参数
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

#神经网络结构参数
#输入图像像素28x28
n_input = 784
#分类数0-9数字
n_classes = 10
#过拟合优化，Dropout概率0.75
dropout = 0.75

#定义输入输出
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
#dropout
keep_prob = tf.placeholder(tf.float32)

#保存模型路径及名称设置
MODEL_SAVE_PATH = "./modelAlexNet"
MODEL_NAME = "model.ckpt"

#定义卷积操作
def conv2d(name, x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x, name=name)

#池化层
def maxpool2d(name, x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
		padding='SAME', name=name)

#规范
def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, 
		beta=0.75, name=name)

#定义网络参数:权重
weights ={
	'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
	'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
	'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
	'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
	'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
	'wd1': tf.Variable(tf.random_normal([2*2*256, 4096])),
	'wd2': tf.Variable(tf.random_normal([4096, 4096])),
	'out': tf.Variable(tf.random_normal([4096, 10]))
}

#定义网络参数:偏置
biases = {
	'bc1': tf.Variable(tf.random_normal([96])),
	'bc2': tf.Variable(tf.random_normal([256])),
	'bc3': tf.Variable(tf.random_normal([384])),
	'bc4': tf.Variable(tf.random_normal([384])),
	'bc5': tf.Variable(tf.random_normal([256])),
	'bd1': tf.Variable(tf.random_normal([4096])),
	'bd2': tf.Variable(tf.random_normal([4096])),
	'out': tf.Variable(tf.random_normal([n_classes]))

}

#定义网络模型
def alex_net(x, weights, biases, dropout):
	#调整数据格式：维度
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	#第一层卷积
	##卷积
	conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
	##下采样
	pool1 = maxpool2d('pool1', conv1, k=2)
	##规范化
	norm1 = norm('norm1', pool1, lsize=4)

	#第二层卷积
	##卷积
	conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
	##最大池化(下采样)
	pool2 = maxpool2d('pool2', conv2, k=2)
	##规范化
	norm2 = norm('norm2', pool2, lsize=4)

	#第三层卷积
	##卷积
	conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
	##下采样
	pool3 = maxpool2d('pool3', conv3, k=2)
	##规范
	norm3 = norm('norm3', pool3, lsize=4)

	#第四层卷积
	##卷积
	conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

	#第五层卷积
	##卷积
	conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
	##下采样
	pool5 = maxpool2d('pool5', conv5, k=2)
	##规范
	norm5 = norm('norm5', pool5, lsize=4)

	#全连接层1
	fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	#dropout
	fc1 = tf.nn.dropout(fc1, dropout)

	#全连接层2
	fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
	fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
	fc2 = tf.nn.relu(fc2)
	#dropout
	fc2 = tf.nn.dropout(fc2, dropout)

	#输出层
	out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
	return out

#构件模型
pred = alex_net(x, weights, biases, keep_prob)
print("==pred==")
print(type(pred))
print(pred)
print("==y==")
print(type(y))
print(y)

#损失函数与优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#评估函数
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#持久化模型
saver = tf.train.Saver()
#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	step = 1
	#开始训练，直到指定训练步数training_iters
	while step * batch_size < training_iters:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		sess.run(optimizer, feed_dict={x: batch_x, y:batch_y, keep_prob: dropout})

		#输出日志
		if step % display_step == 0:
			#计算损失值和准确度
			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
				y: batch_y, keep_prob: 1.})

			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
				"{:.6f}".format(loss) + ", Traning Accuracy= " + \
				"{:.5f}".format(acc))
			#保存当前模型
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

		step += 1
	print("Optimization Finished")
	print("Testing Accuracy: ",\
		sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
			y: mnist.test.labels[:256],
			keep_prob: 1.}))






















