from data import get_train_data, get_test_data
from model import mk_model_for_train
import bert_utils
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

ds = get_train_data(batch_size=32)
for d in ds:
    print(d)
    break

model = mk_model_for_train({"max_position_embeddings": bert_utils.FLAGS.max_seq_length, "model_dir": "./albert_tiny"})
model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer="sgd",
    metrics=['accuracy'],
)

filepath = "./model_{epoch:02d}-{tf_op_layer_start_positions_accuracy:.2f}_{tf_op_layer_end_positions_accuracy:.2f}_{answer_types_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
model.summary()
callbacks = [
    checkpoint,
    TensorBoard()
]

model.fit(ds, steps_per_epoch = 3,epochs=1000, callbacks=callbacks)
