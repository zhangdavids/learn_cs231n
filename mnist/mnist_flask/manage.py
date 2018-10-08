import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, render_template, request

from mnist import models


x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("regression"):
    y1, variables = models.my_regression_model(x)

saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float32")
    y2, variables = models.convolutional(x, keep_prob)

saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y1, feed_dict={x: input,
                                   keep_prob: 1.0}).flatten().tolist()


@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json,dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)

    return jsonify(results=[output1, output2])


@app.route('/home')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.debug = True
    app.run()
