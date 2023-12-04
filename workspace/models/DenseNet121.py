import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
AUTOTUNE = tf.data.AUTOTUNE

def prepare_sample(features):
    image = tf.image.resize(features["image"], size=(224, 224))
    return image, features["category_id"]

def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset

train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*train.tfrec")
test_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*test.tfrec")

batch_size = 2
epochs = 20
steps_per_epoch = 7

train_dataset = get_dataset(train_filenames, batch_size)
test_dataset = get_dataset(test_filenames, batch_size)

input_tensor = tf.keras.layers.Input(shape=(224, 224, 3), name="image")
model = DenseNet121(
    input_tensor=input_tensor, weights=None, classes=91
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_dataset,
    verbose=1,
)

print(history.history)
