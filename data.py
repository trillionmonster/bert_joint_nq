import tqdm
import bert_utils
import tensorflow as tf

eval_records = "nq-test.tfrecords"
# nq_test_file = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl'
train_records_ = 'nq-train.tfrecords'


def get_train_data(batch_size=16):
    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        name_to_features_x = {"input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
                              "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64)
                              }
        name_to_features_y = {
            "start_positions": tf.io.FixedLenFeature([], tf.int64),
            "end_positions": tf.io.FixedLenFeature([], tf.int64),
            "answer_types": tf.io.FixedLenFeature([], tf.int64)
        }
        example_x = tf.io.parse_single_example(serialized=record, features=name_to_features_x)
        example_y = tf.io.parse_single_example(serialized=record, features=name_to_features_y)
        example_y["tf_op_layer_start_positions"] = example_y["start_positions"]
        example_y["tf_op_layer_end_positions"] = example_y["end_positions"]
        # example_y["answer_types"] = tf.one_hot(example_y["answer_types"], 5,dtype=tf.int64)


        return example_x, example_y
               # [tf.one_hot(example_y["start_positions"], 512,dtype=tf.int64), tf.one_hot(example_y["end_positions"], 512,dtype=tf.int64),
               #             tf.one_hot(example_y["answer_types"], 5,dtype=tf.int64)]

    raw_ds = tf.data.TFRecordDataset(train_records_)
    decoded_ds = raw_ds.map(_decode_record)
    ds = decoded_ds.batch(batch_size=batch_size, drop_remainder=False)

    return ds


seq_length = bert_utils.FLAGS.max_seq_length  # config['max_position_embeddings']


def get_test_data(batch_size=16):
    seq_length = bert_utils.FLAGS.max_seq_length  # config['max_position_embeddings']
    name_to_features = {
        "unique_id": tf.io.FixedLenFeature([], tf.int64),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features=name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(serialized=record, features=name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if name != 'unique_id':  # t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = t

        return example

    def _decode_tokens(record):
        return tf.io.parse_single_example(serialized=record,
                                          features={
                                              "unique_id": tf.io.FixedLenFeature([], tf.int64),
                                              "token_map": tf.io.FixedLenFeature([seq_length], tf.int64)
                                          })

    raw_ds = tf.data.TFRecordDataset(eval_records)
    token_map_ds = raw_ds.map(_decode_tokens)
    decoded_ds = raw_ds.map(_decode_record)
    ds = decoded_ds.batch(batch_size=batch_size, drop_remainder=False)

    return ds, token_map_ds
