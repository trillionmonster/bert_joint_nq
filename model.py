import tensorflow as tf
import bert
from bert import BertModelLayer

config_path = "./albert_tiny/albert_config_tiny.json"
checkpoint_path = "./albert_tiny/albert_model.ckpt"


class TDense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                 **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `TDense` layer with "
                            "non-floating point (and non-complex) "
                            "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to "
                             "`TDense` should be defined. "
                             "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.output_size, last_dim],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        super(TDense, self).build(input_shape)

    def call(self, x):
        return tf.matmul(x, self.kernel, transpose_b=True) + self.bias


def get_initializer(initializer_range=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def mk_model_for_train(config):
    model_dir = config["model_dir"]
    seq_len = config['max_position_embeddings']
    # unique_id = tf.keras.Input(shape=(1,), dtype=tf.int64, name='unique_id')
    input_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='segment_ids')

    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="albert")

    sequence_output = l_bert([input_ids, segment_ids])

    pooler_transform = tf.keras.layers.Dense(
        units=bert_params["hidden_size"],
        activation="tanh",
        kernel_initializer=get_initializer(),
        name="pooler_transform")
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    pooled_output = pooler_transform(first_token_tensor)

    logits = TDense(2, name='logits')(sequence_output)
    start_logits, end_logits = tf.split(logits, axis=-1, num_or_size_splits=2, name='split')
    start_logits = tf.squeeze(start_logits, axis=-1, name='start_positions')
    end_logits = tf.squeeze(end_logits, axis=-1, name='end_positions')

    answer_types = tf.keras.layers.Dense(5,activation="softmax",name="answer_types")(pooled_output)
    return tf.keras.Model([input_ for input_ in [ input_ids, segment_ids]
                           if input_ is not None],
                          [ start_logits, end_logits,answer_types],
                          name='bert-baseline')
def mk_model_for_test(hdf5_file,seq_len):
    model_ = tf.keras.models.load_model(hdf5_file,custom_objects = {"BertModelLayer":BertModelLayer})

    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int32,name='unique_id')

    input_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name='segment_ids')
    model_ = model_()([input_ids,segment_ids])
    model = tf.keras.Model([unique_id,input_ids,segment_ids], [ unique_id, model_])
    return model

