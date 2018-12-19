from abc import abstractmethod, ABCMeta
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import h5py

class generator:
    def __init__(self, file, table_name='embeddings'):
        self.file = file
        self.table_name = table_name
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf[self.table_name]:
                yield im


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        assert callable(self.iterator_initializer_func)
        self.iterator_initializer_func(session)


class BaseInputFunction(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, data, batch_size, num_epochs, mode, scope):
        self.data = data
        self.batch_size = batch_size
        self.mode = mode
        self.scope = scope
        self.num_epochs = num_epochs
        self.init_hook = IteratorInitializerHook()

    @abstractmethod
    def _create_placeholders(self):
        """Returns placeholders for input data

        Returns
        -------
        data : tf.placeholder
            Placeholder for training data

        labels : tf.placeholder
            Placeholder for labels
        """

    @abstractmethod
    def _get_feed_dict(self, placeholders):
        """Return feed_dict to initialize placeholders.

        Parameters
        ----------
        placeholders : list of tf.placeholder
            Placeholders to initialize

        Returns
        -------
        feed_dict : dict
            Dictionary with values used to initialize
            passed tf.placeholders.
        """

    def _build_dataset(self, placeholders):
        return tf.data.Dataset.from_tensor_slices(placeholders)

    def __call__(self):
        with tf.name_scope(self.scope):
            # Define placeholders
            placeholders = self._create_placeholders()

            # Build dataset iterator
            dataset = self._build_dataset(placeholders)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.shuffle(buffer_size=10000)
                dataset = dataset.repeat(self.num_epochs)
            dataset = dataset.batch(self.batch_size)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                # skip partial batches
                dataset = dataset.filter(
                    lambda x, y: tf.equal(tf.shape(x)[0], self.batch_size))
            dataset = dataset.prefetch(2)

            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()

            def _init(sess):
                sess.run(iterator.initializer,
                         feed_dict=self._get_feed_dict(placeholders))

            self.init_hook.iterator_initializer_func = _init

        return next_example, next_label


class CorruptedInputDecorator(BaseInputFunction):
    """Corrupts input with noise

    Parameters
    ----------
    input_function : BaseInputFunction
        Input function to wrap.

    noise_factor : float
        Amount of noise to apply.
    """

    def __init__(self, input_function, noise_factor=0.5):
        super().__init__(data=input_function.data,
                         batch_size=input_function.batch_size,
                         num_epochs=input_function.num_epochs,
                         mode=input_function.mode,
                         scope=input_function.scope)
        self.input_function = input_function
        self.noise_factor = noise_factor

    def _create_placeholders(self):
        return self.input_function._create_placeholders()

    def _get_feed_dict(self, placeholders):
        return self.input_function._get_feed_dict(placeholders)

    def _build_dataset(self, placeholders):
        dataset = self.input_function._build_dataset(placeholders)

        def add_noise(input_img, groundtruth_img):
            noise = self.noise_factor * tf.random_normal(input_img.shape.as_list())
            input_corrupted = tf.clip_by_value(tf.add(input_img, noise), 0., 1.)
            return input_corrupted, groundtruth_img

        # run mapping function in parallel
        return dataset.map(add_noise, num_parallel_calls=4)


class MNISTReconstructionInputFunction(BaseInputFunction):
    """MNIST input function to train an autoencoder.

    Parameters
    ----------
    data : tensorflow.examples.tutorials.mnist.input_data
        MNIST dataset.

    mode : int
        Train, eval or prediction mode.

    scope : str
        Name of input function in Tensor board.
    """

    def __init__(self, data, batch_size, num_epochs, mode, scope):
        super().__init__(data=data, batch_size=batch_size,
                         num_epochs=num_epochs, mode=mode, scope=scope)
        self._images_placeholder = None
        self._labels_placeholder = None

    def _create_placeholders(self):
        images_placeholder = tf.placeholder(self.data.dtype, self.data.shape,
                                            name='input_image')
        labels_placeholder = tf.placeholder(self.data.dtype, self.data.shape,
                                            name='reconstruct_image')
        return images_placeholder, labels_placeholder

    def _get_feed_dict(self, placeholders):
        assert len(placeholders) == 2
        return dict(zip(placeholders, [self.data, self.data]))


class MNISTReconstructionDataset:
    """MNIST data set for learning an unsupervised autoencoder

    Parameters
    ----------
    data_dir : str
        Path to directory to write data to.

    noise_factor : float
        The amount of noise to apply. If non-zero, a denoising
        autoencoder will be trained.
    """

    def __init__(self, data_dir, noise_factor=0.0, number_of_tokens=None):
        #self.mnist = mnist_data.read_data_sets(data_dir, one_hot=False)
        self.noise_factor = noise_factor
        self.data_dir = data_dir
        self.number_of_tokens = number_of_tokens
        #self.feature_columns = tf.feature_column.numeric_column('x', shape=(784,))

    def _input_fn_corrupt(self, data, batch_size, num_epochs, mode, scope):
        f = MNISTReconstructionInputFunction(data, batch_size, num_epochs,
                                             mode, scope)
        if self.noise_factor > 0:
            return CorruptedInputDecorator(f, noise_factor=self.noise_factor)
        return f
    def _slice_number_of_tokens(self, data):
        if self.number_of_tokens is None:
            return data
        return  data.take(indices=range(0, self.number_of_tokens), axis=1)
    def get_train_input_fn(self, batch_size, num_epochs):
        #ds = self._read_dataset(os.path.join(self.data_dir, "train_question_embeddings.hdf5"))
        self.train = self.load_embeddings(os.path.join(self.data_dir, "train_embeddings.hdf5"))
        self.train = self._slice_number_of_tokens(self.train)
        #self.train = self.mnist.train.images
        return self._input_fn_corrupt(self.train, batch_size, num_epochs,
                                      tf.estimator.ModeKeys.TRAIN,
                                      'training_data')

    def get_eval_input_fn(self, batch_size):
        self.eval = self.load_embeddings(os.path.join(self.data_dir, "test_embeddings.hdf5"))
        self.eval = self._slice_number_of_tokens(self.eval)
        #self.eval = self.mnist.validation.images
        return self._input_fn_corrupt(self.eval, batch_size, None,
                                      tf.estimator.ModeKeys.EVAL,
                                      'validation_data')

    def get_test_input_fn(self, batch_size):
        self.test = self.load_embeddings(os.path.join(self.data_dir, "embeddings.hdf5"))
        self.test = self._slice_number_of_tokens(self.test)
        #self.test = self.mnist.test.images
        return self._input_fn_corrupt(self.test, batch_size, None,
                                      tf.estimator.ModeKeys.PREDICT,
                                      'test_data')

    def _read_dataset(self, file_path, table_name='embeddings', data_type=tf.float32):
        ds = tf.data.Dataset.from_generator(
            generator(file_path, table_name),
            data_type,
            tf.TensorShape([150, 1024, ]))
        return ds

    def load_embeddings(self, infile_to_get):
        with h5py.File(infile_to_get, 'r') as fin:
            document_embeddings = fin['embeddings'][...]
        return document_embeddings