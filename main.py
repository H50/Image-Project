
import os.path
import math
import tensorflow as tf
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import helper


# parameters
KEEP_PROB = 0.5
LEARNING_RATE = 0.0005
epochs = 20
batch_size = 4

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    1-param sess: TensorFlow Session
    2-param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    return:Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    graph=tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep, vgg_layer3, vgg_layer4, vgg_layer7



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    1-param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    2-param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    3-param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    4-param num_classes: Number of classes to classify
    return: The Tensor for the last layer of output
    """
 
    cov_1x1 = tf.layers.conv2d(vgg_layer7_out,num_classes, 1, padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev = 0.01))
    output = tf.layers.conv2d_transpose(cov_1x1, num_classes, 4, strides=(2,  2), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev=0.01))
    layer_4_cov_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev=0.01))
    output = tf.add(output, layer_4_cov_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, strides=(2,  2), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev=0.01))
    layer_3_cov_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev=0.01))
    output = tf.add(output, layer_3_cov_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, strides=(8,  8), padding='same', kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer = tf.random_normal_initializer(stddev=0.01))
    
    return output


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    1-param nn_last_layer: TF Tensor of the last layer in the neural network
    2-param correct_label: TF Placeholder for the correct label image
    3-param learning_rate: TF Placeholder for the learning rate
    4-param num_classes: Number of classes to classify
    return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)
    return logits, training_operation, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    1-param sess: TF Session
    2-param epochs: Number of epochs
    3-param batch_size: Batch size
    4-param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    5-param train_op: TF Operation to train the neural network
    6-param cross_entropy_loss: TF Tensor for the amount of loss
    7-param input_image: TF Placeholder for input images
    8-param correct_label: TF Placeholder for label images
    9-param keep_prob: TF Placeholder for dropout keep probability
    10-param learning_rate: TF Placeholder for learning rate
    """
  
    sess.run(tf.global_variables_initializer())
    print("Training...\n")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        index=0
        for image,label in get_batches_fn(batch_size):
            loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label,keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE})        
            index=index+1
            print(index,"/",math.ceil(289/batch_size)," Loss = {:.3f}                    ".format(loss),end="\r")
        print()      



def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    correct_label = tf.placeholder(tf.int32, shape=[None, None, None, num_classes])
    learning_rate = tf.placeholder(tf.float32)
    
    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        image_input, keep, vgg_layer3, vgg_layer4, vgg_layer7=load_vgg(sess,vgg_path)
        output = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
        saver = tf.train.Saver()
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,correct_label, keep, learning_rate)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep, image_input)
        saver.save(sess, './road_semantic_segmentation')
        print("Model saved")

        
run()

