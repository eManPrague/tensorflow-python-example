import tensorflow as tf
import numpy as np
import cv2
from os import listdir
from os.path import isfile, isdir, join
import os


def path_current():
    return os.path.dirname(os.path.realpath(__file__))


def weight_variable(shape):
    with tf.name_scope("weight"):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
    with tf.name_scope("bias"):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    flatten_image = image.ravel()
    normalized_image = flatten_image / 255
    return np.float32(normalized_image)


def load_dataset_hot_it(path, subfolder):
    my_path = join(path, subfolder)
    dirs = [f for f in listdir(my_path) if isdir(join(my_path, f))]

    data = np.empty(shape=[0, 32*32*1], dtype=np.float32)
    labels = np.empty(shape=[0, len(dirs)], dtype=np.float32)
    length = 0

    for dir in dirs:
        label, label_num = os.path.splitext(dir)
        dir_path = join(my_path, dir)
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
        for file in files:
            _, extension = os.path.splitext(file)
            if extension != '.png':
                continue
            features = load_image(join(dir_path, file))
            data = np.vstack([data, features])

            index = int(label_num[1:])
            label_temp = np.zeros((1, len(dirs)))
            label_temp[0][index] = 1
            labels = np.vstack([labels, label_temp])

            length += 1

    return data, labels, length


def freeze(checkpoint_path):
    with tf.Session() as sess:
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
        saver.restore(sess, checkpoint_path)

        # Get the input and output tensors needed for toco
        input_tensor = sess.graph.get_tensor_by_name("input_tensor:0")
        input_tensor.set_shape([1, 1024])
        out_tensor = sess.graph.get_tensor_by_name("softmax_tensor:0")
        out_tensor.set_shape([1, 2])

        # Pass the output tensor and freeze graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names=["softmax_tensor"])

    tflite_model = tf.contrib.lite.toco_convert(frozen_graph_def, [input_tensor], [out_tensor])
    open("model.tflite", "wb").write(tflite_model)
    print("Frozen model saved")


def define_and_train():

    image_size = 32
    classes = 2
    input_layer_name = "input_tensor"
    output_layer_name = "softmax_tensor"

    train_dir = "tmp"
    learning_rate = 0.01
    batch_size = 32
    train_steps = 3000
    logging_step = 200
    checkpoint_step = 500

    # Define graph
    ############################################################

    input_layer = tf.placeholder(tf.float32, shape=[None, image_size * image_size], name=input_layer_name)
    input_image = tf.reshape(input_layer, shape=[-1, image_size, image_size, 1])

    # 1 Convolution layer
    conv1_w = weight_variable([5, 5, 1, 32])
    conv1_b = bias_variable([32])
    conv1 = tf.nn.relu(conv2d(input_image, conv1_w) + conv1_b)
    pool1 = max_pool_2x2(conv1)

    # 2 Convolution layer
    conv2_w = weight_variable([5, 5, 32, 64])
    conv2_b = bias_variable([64])
    conv2 = tf.nn.relu(conv2d(pool1, conv2_w) + conv2_b)
    pool2 = max_pool_2x2(conv2)

    # Flatten
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

    # 3 Fully connected layer
    full_layer1_w = weight_variable([8 * 8 * 64, 1024])
    full_layer1_b = bias_variable([1024])
    full_layer1 = tf.nn.relu(tf.matmul(pool2_flat, full_layer1_w) + full_layer1_b)

    # 4 Fully connected layer
    full_layer2_w = weight_variable([1024, classes])
    full_layer2_b = bias_variable([classes])
    full_layer2 = tf.matmul(full_layer1, full_layer2_w) + full_layer2_b

    # Output
    output = tf.nn.softmax(full_layer2, name=output_layer_name)  # softmax output
    pred = tf.argmax(output, axis=1)  # predictions

    # Placeholders used for training
    output_true = tf.placeholder(tf.float32, shape=[None, classes])
    pred_true = tf.argmax(output_true, axis=1)


    # Calculate loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=output_true, logits=full_layer2))
    # Configure training operation
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # Add evaluation metrics
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, pred_true), tf.float32))


    # Training
    ############################################################

    train_data, train_labels, train_length = load_dataset_hot_it(path_current(), "dataset/train")
    eval_data, eval_labels, _ = load_dataset_hot_it(path_current(), "dataset/test")

    # Initialize variables (assign default values..)
    init = tf.global_variables_initializer()
    # Initialize saver
    saver = tf.train.Saver()

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(init)
        summary_writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())

        for step in range(train_steps+1):
            # Get random batch
            idx = np.random.randint(train_length, size=batch_size)
            batchX = train_data[idx, :]
            batchY = train_labels[idx, :]

            # Run the optimizer
            _, train_loss, train_accuracy, summary = session.run(
                [optimizer, loss, accuracy, merged_summary_op],
                feed_dict={input_layer: batchX,
                           output_true: batchY}
            )
            # Add summary for tensorboard
            summary_writer.add_summary(summary, step)

            # Test training
            if step % logging_step == 0:
                test_loss, test_accuracy = session.run(
                    [loss, accuracy],
                    feed_dict={input_layer: eval_data,
                               output_true: eval_labels}
                )

                print("Step {0:d}: Loss = {1:.4f}, Accuracy = {2:.3f}".format(step, test_loss, test_accuracy))

            # Save checkpoint
            if step % checkpoint_step == 0:
                saver.save(session, path_current() + "/tmp/model.ckpt", global_step=step)

        print("Training finished")


def main():
    define_and_train()

    checkpoint_path = "tmp/model.ckpt-3000"
    freeze(checkpoint_path)


if __name__ == "__main__":
    main()
