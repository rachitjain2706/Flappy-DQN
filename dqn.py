import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)


def createNetwork():
    # Assigning network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])


def flappy():
    sess = tf.InteractiveSession()
    createNetwork() # s, readout, h_fc1 = createNetwork()


def main():
    flappy()


if __name__ == "__main__":
    main()