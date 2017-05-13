import tensorflow as tf

class RBM(object):
    def __init__(self, num_visble, num_hidden):
        self.dim = (num_visble, num_hidden)
        self.num_visble = num_visble
        self.num_hidden = num_hidden
        self.create_placeholder()
        self.add_model()
        self.optimizer = self.train(self.input)

    def create_placeholder(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.num_visble],
                                    name="input")
    
    def add_model(self):
        with tf.variable_scope("model"):
            self.weights = tf.get_variable(name="weights", shape=self.dim,
                                           dtype=tf.float32)
            self.hbias = tf.get_variable(name="hbias", shape=[self.num_hidden],
                                         dtype=tf.float32)
            self.vbias = tf.get_variable(name="vbias", shape=[self.num_visble],
                                         dtype=tf.float32)
            self.prev_gw = tf.get_variable(name="prev_weights", shape=self.dim,
                                           dtype=tf.float32)
            self.prev_gbh = tf.get_variable(name="prev_hbias", shape=[self.num_hidden],
                                         dtype=tf.float32)
            self.prev_gbv = tf.get_variable(name="prev_vbias", shape=[self.num_visble],
                                         dtype=tf.float32)

    def sample_hidden(self, vis):
        activations = tf.nn.sigmoid(tf.matmul(vis, self.weights) + self.hbias)
        h1_sample = tf.sign(activations - tf.random_uniform(tf.shape(activations)))
        return activations, h1_sample

    def sample_visible(self, hid, k=10):
        activations = tf.nn.sigmoid(tf.matmul(hid, tf.transpose(self.weights)) + self.vbias)
        k_ones = tf.ones((1,k))
        partition = \
            tf.matmul(tf.reshape(tf.reduce_sum(tf.reshape(activations, (-1, k)), 1),
                                 (-1, 1)) , k_ones)
       
        activations = activations / tf.reshape(partition , tf.shape(activations))
        v1_sample = tf.sign(activations - tf.random_uniform(tf.shape(activations)))
       
        return activations, v1_sample

    def contrastive_divergence(self, v1):
        h1, _ = self.sample_hidden(v1)
        v2, v2a = self.sample_visible(h1)
        h2, h2a = self.sample_hidden(v2)

        return (v1, h1, v2, v2a, h2, h2a)

    def gradient(self, v1, h1, v2, h2a):
        gw = tf.matmul(tf.transpose(v1), h1) - tf.matmul(tf.transpose(v2), h2a)
        gbv = tf.reduce_mean(v1 - v2, 0) 
        gbh = tf.reduce_mean(h1 - h2a, 0)
        return [gw, gbv, gbh]
    
    def train(self, vis,  w_lr=0.000021, v_lr=0.000025,
                h_lr=0.000025, decay=0.0000, momentum=0.0):
        v1, h1, v2, v2a, h2, h2a = self.contrastive_divergence(vis)
        self.predict = v2a
        gw, gbv, gbh = self.gradient(v1, h1, v2, h2a)
        update_w = tf.assign(self.weights, 
                             self.weights + momentum * self.prev_gw + w_lr * gw)
        update_bh = tf.assign(self.hbias,
                              self.hbias + momentum * self.prev_gbh + h_lr * gbh)
        update_bv = tf.assign(self.vbias,
                              self.vbias + momentum * self.prev_gbv + v_lr * gbv)
        print(gbv.get_shape(), self.prev_gbv.get_shape())
        update_prev_gw = tf.assign(self.prev_gw, gw)
        update_prev_gbh = tf.assign(self.prev_gbh, gbh)
        update_prev_gbv = tf.assign(self.prev_gbv, gbv)
        optimizer = (update_w, update_bh, update_bv, update_prev_gw,
                     update_prev_gbh, update_prev_gbv)
        return optimizer

if __name__ == "__main__":
    # for test model
    RBM(300000, 100)


