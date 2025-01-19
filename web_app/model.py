import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras

@keras.utils.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(0.1)
        self.dropout2 = keras.layers.Dropout(0.1)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
        })
        return config

@keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.token_emb = layers.Dense(embed_dim)  # Project input features to embed_dim
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]  # Length of the sequence (number of timesteps)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)  # Map the input sequence to the embed_dim space
        return x + positions  # Add positional information to the input
    
# Define the model
def build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_classes, maxlen):
    inputs = layers.Input(shape=input_shape)
    
    # Embedding input sequence
    x = PositionalEmbedding(maxlen, embed_dim)(inputs)
    
    # Transformer Block
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x, training = True)
    
    # Global average pooling layer
    x = layers.GlobalAveragePooling1D()(x)
    
    # Fully connected layer
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer (for classification, use softmax activation)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

if __name__ == "__main__":
    # Load preprocessed features and labels
    X_train = np.load('train_data_features.npy')
    y_train = np.load('train_data_labels.npy')
    X_test = np.load('test_data_features.npy')
    y_test = np.load('test_data_labels.npy')

    X_train = X_train.reshape((X_train.shape[0], -1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], -1, X_test.shape[1]))

    # Define parameters
    embed_dim = 64  # Embedding size for each token
    num_heads = 4   # Number of attention heads
    ff_dim = 128    # Hidden layer size in feed-forward network inside transformer
    maxlen = X_train.shape[1]  # Maximum sequence length
    num_classes = 6  # Number of emotion classes

    # Build the transformer model
    input_shape = (maxlen, X_train.shape[2])  # (sequence_length, feature_dimension)
    model = build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_classes, maxlen)

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split= 0.2)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')

    model.save('emotion_recognition_model.keras')
