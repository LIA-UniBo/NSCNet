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
        self.margin_cosface = self.sin_margin * margin

        self.output_units = output_units
        self.output_regularizer = output_regularizer

    def build(self, input_shape):

        self.custom_weights = self.add_weight(name='custom_weights',
                                              shape=(input_shape[0][-1], self.output_units),
                                              initializer='glorot_uniform',
                                              trainable=True,
                                              regularizer=self.output_regularizer)

    def call(self, inputs):

        features = inputs[0]
        labels = inputs[1]
        # features and weights normalization
        features_norm = tf.math.reduce_euclidean_norm(features, axis=1, keepdims=True)
        normalized_features = tf.math.divide(features, features_norm)

        weights_norm = tf.math.reduce_euclidean_norm(self.custom_weights, axis=0, keepdims=True)
        normalized_weights = tf.math.divide(self.custom_weights, weights_norm)

        # dot product that returns the so called "cos(θ)" in the paper
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
        cos_theta = tf.tensordot(normalized_features, normalized_weights, axes=1)

        # Compute scale * cos(θ + margin)
        """
        Use the trigonometric function
        sin^2(θ) = 1 - cos^2(θ)
        cos(x1 + x2) = cos(x1) * cos(x2) - sin(x1) * sin(x2) 
        """
        cos_theta_squared = tf.math.square(cos_theta)
        sin_theta_squared = tf.math.subtract(1.0, cos_theta_squared)
        sin_theta = tf.math.sqrt(sin_theta_squared)

        cos_theta_arcface = tf.math.subtract(
            tf.math.multiply(cos_theta, self.cos_margin),
            tf.math.multiply(sin_theta, self.sin_margin)
        )
        cos_theta_arcface_scaled = tf.math.multiply(self.scale, cos_theta_arcface)

        # Constraint θ + margin < π
        """
        θ is the angle between two vectors.
        By definition, the angle between two vectors is defined as the minimum non-negative angle 
        separating their directions. Therefore, the allowed values are in the range [0, π].
        
        Nevertheless, when the margin is added we could exceed π, and this is something we want to avoid, 
        because the application of the margin would be counteproductive. When this happens, then it is applied 
        the CosFace loss, which empirically seems to work better.
        
        References:
        https://github.com/ronghuaiyang/arcface-pytorch/issues/24#issue-428144388
        https://github.com/ronghuaiyang/arcface-pytorch/issues/24#issuecomment-510078581 
        """
        arcface_condition_function = cos_theta - self.threshold
        arcface_condition_mask = tf.cast(tf.nn.relu(arcface_condition_function), dtype=tf.bool)

        cos_theta_cosface = tf.math.subtract(cos_theta, self.margin_cosface)
        cos_theta_cosface_scaled = tf.math.multiply(self.scale, cos_theta_cosface)

        cos_theta_conditioned = tf.where(arcface_condition_mask, cos_theta_arcface_scaled, cos_theta_cosface_scaled)

        # Create a mask for applying the arcface loss only when required
        gt_mask = tf.one_hot(labels, depth=self.output_units)
        gt_inverted_mask = tf.math.subtract(1.0, gt_mask)

        # Scale the original logits obtained by the dot product operation
        # This is necessary for avoiding to apply the arcFace to logits that are not
        # referring to the ground truth class
        cos_theta_scaled = tf.math.multiply(self.scale, cos_theta)

        # Compute the final layer output by performing the following operations:
        # 1) keep the original (scaled) logits for those that are NOT referring to the ground truth class
        # 2) use the arcFace otherwise
        # 3) sum the results of 1) and 2)
        # 4) apply the softmax
        output = tf.math.add(
            tf.math.multiply(cos_theta_scaled, gt_inverted_mask),
            tf.math.multiply(cos_theta_conditioned, gt_mask)
        )

        output = tf.keras.activations.softmax(output)

        return output

