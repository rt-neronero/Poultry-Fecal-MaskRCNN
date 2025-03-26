import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

class ParallelModel(KM.Model):
    def __init__(self, keras_model, gpu_count):
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs, outputs=merged_outputs)

    def __getattribute__(self, attrname):
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        input_slices = {name: tf.split(x, self.gpu_count) for name, x in zip(self.inner_model.input_names, self.inner_model.inputs)}
        output_names = self.inner_model.output_names
        outputs_all = [[] for _ in range(len(self.inner_model.outputs))]

        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    zipped_inputs = zip(self.inner_model.input_names, self.inner_model.inputs)
                    inputs = [KL.Lambda(lambda s: input_slices[name][i], output_shape=lambda s: (None,) + s[1:])(tensor) for name, tensor in zipped_inputs]
                    outputs = self.inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                if K.int_shape(outputs[0]) == ():
                    m = KL.Lambda(lambda o: tf.add_n(o) / len(outputs), name=name)(outputs)
                else:
                    m = KL.Concatenate(axis=0, name=name)(outputs)
                merged.append(m)
        return merged

def build_model(x_train, num_classes):
    inputs = KL.Input(shape=x_train.shape[1:], name="input_image")
    x = KL.Conv2D(32, (3, 3), activation='relu', padding="same", name="conv1")(inputs)
    x = KL.Conv2D(64, (3, 3), activation='relu', padding="same", name="conv2")(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = KL.Flatten(name="flat1")(x)
    x = KL.Dense(128, activation='relu', name="dense1")(x)
    x = KL.Dense(num_classes, activation='softmax', name="dense2")(x)
    return KM.Model(inputs, x, "digit_classifier_model")

if __name__ == "__main__":
    GPU_COUNT = 2
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.expand_dims(x_train, -1).astype('float32') / 255
    x_test = tf.expand_dims(x_test, -1).astype('float32') / 255

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    datagen = ImageDataGenerator()
    model = build_model(x_train, 10)
    model = ParallelModel(model, GPU_COUNT)
    
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, clipnorm=5.0)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=50, epochs=10, verbose=1,
        validation_data=(x_test, y_test)
    )
