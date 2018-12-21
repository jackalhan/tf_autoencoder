import tensorflow as tf

from .layers import fully_connected_autoencoder, convolutional_autoencoder



def pairwise_euclidean_distances(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of B.
    Args:
      A,    [m,d] matrix
      B,    [n,d] matrix
    Returns:
      D,    [m,n] matrix of pairwise distances
    """
    with tf.variable_scope('pairwise_euclidean_dist'):
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 2)
        nb = tf.reduce_sum(tf.square(B), 2)

        # na as a row and nb as a co"lumn vectors
        na = tf.expand_dims(na, 2)
        nb = tf.expand_dims(nb, 1)

        # return pairwise euclidead difference matrix
        D = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0)),axis=1), axis=1))
    return D

def distance_loss(embeddings_origin, embedding_generated):
    """Build the triplet loss over a batch of question and paragraph embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        ....

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix between question and paragraphs
    pairwise_dist = pairwise_euclidean_distances(embeddings_origin, embedding_generated)

    loss = tf.reduce_mean(pairwise_dist)

    return loss

def _create_estimator_spec_from_logits(labels, logits, learning_rate, mode, hparams=None):
    """Add loss function and create estimator spec.

    Parameters
    ----------
    labels : tf.Tensor
        Tenor holding the data to reconstruct.
    logits : tf.Tensor
        Tenor holding the reconstructed data.
    learning_rate : float
        Learning rate.
    mode : tf.estimator.ModeKeys
        The mode of the model.

    Returns
    -------
    spec : tf.estimator.EstimatorSpec
        Specification of the model.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    probs = tf.nn.sigmoid(logits)

    predictions = {"prediction": probs}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # paragraph_embedding_mean_norm = tf.reduce_mean(tf.norm(paragraph, axis=1))
    if hparams.loss == 'custom_distance_loss':
        labels = tf.nn.l2_normalize(labels, axis=2)
        logits = tf.nn.l2_normalize(logits, axis=2)
        _loss = distance_loss(labels, logits)
        tf.losses.add_loss(_loss)
        total_loss = tf.losses.get_total_loss()
    else:
        tf.losses.sigmoid_cross_entropy(labels, logits)
        total_loss = tf.losses.get_total_loss(add_regularization_losses=is_training)

    #raise ValueError("Loss strategy not recognized: {}".format(hparams.loss))

    train_op = None
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            optimizer="Adam",
            learning_rate=learning_rate,
            #learning_rate_decay_fn=lambda lr, gs: tf.train.exponential_decay(lr, gs, 1000, 0.96, staircase=True),
            global_step=tf.train.get_global_step(),
            summaries=["learning_rate", "global_gradient_norm"])

        # Add histograms for trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    elif mode == tf.estimator.ModeKeys.EVAL:
        if hparams.loss == 'custom_distance_loss':
            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(
                    tf.cast(labels, tf.float64), tf.cast(probs, tf.float64))
            }
        else:
            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(
                    tf.cast(labels, tf.float64), tf.cast(probs, tf.float64))
            }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


class AutoEncoder(tf.estimator.Estimator):
    """An Autoencoder estimator with fully connected layers.

    Parameters
    ----------
    hidden_units : list of int
        Number of units in each hidden layer.
    activation_fn : callable|None
        Activation function to use.
    dropout : float|None
         Percentage of nodes to remain activate in each layer,
         or `None` to disable dropout.
    weight_decay : float|None
        Amount of regularization to use on the weights
        (excludes biases).
    learning_rate : float
        Learning rate.
    model_dir : str
        Directory where outputs (checkpoints, event files, etc.)
        are written to.
    config : RunConfig
        Information about the execution environment.
    """

    def __init__(self, hidden_units, activation_fn=tf.nn.relu,
                 dropout=None, weight_decay=1e-5, learning_rate=0.001, model_dir=None, config=None):
        def _model_fn(features, labels, mode):
            # Define model's architecture
            logits = fully_connected_autoencoder(inputs=features,
                                                 hidden_units=hidden_units,
                                                 activation_fn=activation_fn,
                                                 dropout=dropout,
                                                 weight_decay=weight_decay,
                                                 mode=mode)
            return _create_estimator_spec_from_logits(
                labels=labels,
                logits=logits,
                learning_rate=learning_rate,
                mode=mode)

        super(AutoEncoder, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config)


class ConvolutionalAutoencoder(tf.estimator.Estimator):
    """An Autoencoder estimator with 2D convolutions.

    Parameters
    ----------
    num_filters : list of int
        Number of filters in each convolutional block.
    activation_fn : callable|None
        Activation function to use.
    dropout : float|None
         Percentage of nodes to remain activate in each layer,
         or `None` to disable dropout.
    weight_decay : float|None
        Amount of regularization to use on the weights
        (excludes biases).
    learning_rate : float
        Learning rate.
    model_dir : str
        Directory where outputs (checkpoints, event files, etc.)
        are written to.
    config : RunConfig
        Information about the execution environment.
    """

    def __init__(self, num_filters, activation_fn=tf.nn.relu,
                 dropout=None, weight_decay=1e-5, learning_rate=0.001, model_dir=None, config=None, hparams=None):
        def _model_fn(features, labels, mode):
            # Define model's architecture
            logits = convolutional_autoencoder(inputs=features,
                                               num_filters=num_filters,
                                               activation_fn=activation_fn,
                                               dropout=dropout,
                                               weight_decay=weight_decay,
                                               mode=mode,
                                               hparams=hparams)
            return _create_estimator_spec_from_logits(
                labels=labels,
                logits=logits,
                learning_rate=learning_rate,
                mode=mode,
                hparams=hparams)

        super(ConvolutionalAutoencoder, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config)
