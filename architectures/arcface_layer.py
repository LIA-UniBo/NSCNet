import tensorflow as tf
import math


class ArcFace(tf.keras.layers.Layer):
    """
    Original paper: https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(self, output_units, output_regularizer=None, margin=0.5, scale=30.0, **kwargs):
        super(ArcFace, self).__init__(**kwargs)

        self.scale = scale
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.margin_cosface = math.sin(math.pi - margin) * margin

        self.output_units = output_units
        self.output_regularizer = output_regularizer

    def build(self, input_shape):

        self.custom_weights = self.add_weight(name='custom_weights',
                                 shape=(input_shape[-1], self.output_units),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.output_regularizer)
        print()

    def call(self, features, labels):

        # features and weights normalization
        features_norm = tf.norm(features, axis=1, keepdims=True)
        normalized_features = tf.div(features, features_norm)

        weights_norm = tf.norm(self.custom_weights, axis=0, keepdims=True)
        normalized_weights = tf.div(self.custom_weights, weights_norm)

        # dot product that returns the so "cos(θ)" in the paper
        # TODO: move the following comment to the project relation
        """
        The dot product between two vectors can be written as
        
        x ⋅ y = ||x|| * ||y|| * cos(θ)
        
        By applying the normalization as shown in the above lines, we
        can consider:
        
        ||x|| = 1
        ||y|| = 1
        
        Therefore, the final expression is:
        
        x ⋅ y = cos(θ)
        """
        cos_theta = normalized_features @ normalized_weights

        # Compute cos(θ + margin)
        # TODO: move the following comment to the project relation
        """
        Use the trigonometric function
        sin^2(θ) = 1 - cos^2(θ)
        cos(x1 + x2) = cos(x1) * cos(x2) - sin(x1) * sin(x2) 
        """
        cos_theta_squared = tf.square(cos_theta)
        sin_theta_squared = tf.subtract(1.0, cos_theta_squared)
        sin_theta = tf.sqrt(sin_theta_squared)

        cos_theta_margin = tf.subtract(
            tf.multiply(cos_theta, self.cos_margin),
            tf.multiply(sin_theta, self.sin_margin)
        )

        # Constraint θ + margin < π
        # TODO: move the following comment to the project relation
        """
        θ is the angle between two vectors.
        By definition, this he angle between two vectors is defined as the minimum non-negative angle 
        separating their directions. Therefore, the allowed values are in the range [0, π].
        
        Nevertheless, when the margin is added we could exceed π, and this is something we want to avoid, 
        because the application of the margin would be counteproductive. When this happens, then it is applied 
        the CosFace loss, which seems to work better.
        
        References:
        https://github.com/ronghuaiyang/arcface-pytorch/issues/24#issue-428144388
        https://github.com/ronghuaiyang/arcface-pytorch/issues/24#issuecomment-510078581 
        """
        condition_function = cos_theta - self.threshold
        condition = tf.cast(tf.relu(condition_function), dtype=tf.bool)

        cos_face = tf.multiply(self.scale, tf.subtract(cos_theta, self.margin_cosface))

        cos_theta_margin_conditioned = tf.where(condition, cos_theta_margin, cos_face)

        # Create a mask for applying the arcface loss only when required
        mask = tf.one_hot(labels, depth=self.output_units)
        inverted_mask = tf.subtract(1.0, mask)

        # Scale all the cos_theta
        cos_theta_scaled = tf.multiply(self.scale, cos_theta)

        # Compute the final layer output
        output = tf.add(
            tf.multiply(cos_theta_scaled, inverted_mask),
            tf.multiply(cos_theta_margin_conditioned, mask)
        )

        output = tf.keras.activations.softmax(output)

        return output












        return 0