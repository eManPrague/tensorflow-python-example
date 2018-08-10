import tensorflow as tf
import numpy as np
import cv2
from os import listdir
from os.path import isfile, isdir, join
import os


def path_current():
    return os.path.dirname(os.path.realpath(__file__))

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    flatten_image = image.ravel()
    normalized_image = flatten_image / 255
    return np.float32(normalized_image)


def load_dataset(path, subfolder):
    my_path = join(path, subfolder)
    dirs = [f for f in listdir(my_path) if isdir(join(my_path, f))]

    data = np.empty(shape=[0, 32*32*1], dtype=np.float32)
    labels = np.array([], dtype=np.int32)

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
            labels = np.append(labels, int(label_num[1:]))

    return data, labels


def freeze(checkpoint_path):
    with tf.Session() as sess:
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
        saver.restore(sess, checkpoint_path)

        # Get the input and output tensors needed for toco.
        input_tensor = sess.graph.get_tensor_by_name("input_tensor:0")
        input_tensor.set_shape([1, 1024])
        out_tensor = sess.graph.get_tensor_by_name("softmax_tensor:0")
        out_tensor.set_shape([1, 2])

        # Pass the output tensor and freeze graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names=["softmax_tensor"])

    tflite_model = tf.contrib.lite.toco_convert(frozen_graph_def, [input_tensor], [out_tensor])
    open("model.tflite", "wb").write(tflite_model)


def CNNModel(features, labels, mode):
    classes = 2

    input = tf.placeholder_with_default(features["image"], [None, 1024], name="input_tensor")
    input_layer = tf.reshape(input, shape=[-1, 32, 32, 1])

    # 1 Convolution layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="SAME",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 2 Convolution layer
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="SAME",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

    # 3 Fully connected layer
    full_layer1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # 4 Fully connected layer
    full_layer2 = tf.layers.dense(inputs=full_layer1, units=classes)

    predictions = {
        # Generate predictions
        "classes": tf.argmax(input=full_layer2, axis=1),
        # Add `softmax_tensor` to the graph
        "probabilities": tf.nn.softmax(full_layer2, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=full_layer2)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    train_data, train_labels = load_dataset(path_current(), "dataset/train")
    eval_data, eval_labels = load_dataset(path_current(), "dataset/test")

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=CNNModel, model_dir=join(path_current(), "tmp"))

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": train_data},
        y=train_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Freeze graph and generate .tflite model
    checkpoint_path = "tmp/model.ckpt-2000"
    freeze(checkpoint_path)


if __name__ == "__main__":
    main()













