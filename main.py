import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    return graph.get_tensor_by_name(vgg_input_tensor_name), graph.get_tensor_by_name(vgg_keep_prob_tensor_name), graph.get_tensor_by_name(vgg_layer3_out_tensor_name), graph.get_tensor_by_name(vgg_layer4_out_tensor_name), graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # return None, None, None, None, None
tests.test_load_vgg(load_vgg, tf)


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
    # vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    # vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    kinit = tf.truncated_normal_initializer(stddev=0.01)
    conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=kinit)

    up2x = tf.layers.conv2d_transpose(conv1x1, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=kinit)
    # TODO: Implement function


    vgg_layer4_conv1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=kinit)
    skip1 = tf.add(up2x, vgg_layer4_conv1x1)
    up2xagain =  tf.layers.conv2d_transpose(skip1, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=kinit)

    vgg_layer3_conv1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=kinit)
    skip2 = tf.add(up2xagain, vgg_layer3_conv1x1)
    up8x = tf.layers.conv2d_transpose(skip2, num_classes, 16, 8, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=kinit)
    #up8x = tf.Print(up8x, [tf.shape(up8x)])
    return up8x
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)
    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    for epoch in tqdm(range(epochs)):
        for image,label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 0.5})
            print("loss {}".format(loss))
            # TODO: Implement function
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    correct_label = tf.placeholder(tf.float32, None)
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        input, keep_prob, layer3, layer4, layer7 =  load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cle = optimize(output, correct_label, 0.0001, num_classes)
        saver = tf.train.Saver()
        train_nn(sess, 50, 16, get_batches_fn, train_op, cle, input, correct_label, keep_prob, 0.001)
        saver.save(sess,'out')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input)
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
