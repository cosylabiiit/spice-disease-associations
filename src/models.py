from keras.layers import Input, Embedding, Convolution1D, Dropout, Dense, Concatenate, GlobalMaxPool1D
from keras.layers import merge
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam


def sentence_cnn(sequence_length, emb_weights, filter_sizes, nb_filters, num_hidden, dropout_prob, l2_reg):
    """
    Constructs a 1D Convolutional Neural Network utilising sentence features towards classifiation tasks.

    Args:
        sequence_length - Length of the longest sentence.
        emb_weights - Pre-trained embedding weights.
        filter_sizes - Widths of filters (type: list).
        nb_filters - Number of filters for each filter size.
        num_hidden - Number of neurons in hidden layer.
        dropout_prob - Value of dropout to use
        l2_reg - Value of l2 based regularisation.

    Returns:
        Compiled model with Categorical Cross-Entropy loss and Adam optimizer.

    """

    model_input = Input(shape=(sequence_length,),
                        dtype='int32', name='Sparse_Input')
    embedding = Embedding(emb_weights.shape[0], emb_weights.shape[1], input_length=sequence_length,
                          weights=[emb_weights],
                          trainable=True, name='Word_Embeddings')(model_input)
    conv_blocks = []
    for fs in filter_sizes:
        conv = Convolution1D(filters=nb_filters,
                             kernel_size=fs,
                             padding="valid",
                             activation="relu",
                             strides=1)(embedding)
        conv = GlobalMaxPool1D()(conv)
        conv_blocks.append(conv)

    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dropout(dropout_prob)(z)
    z = Dense(num_hidden, activation="relu", kernel_regularizer=l2(l2_reg))(z)

    model_output = Dense(3, activation="softmax")(z)
    model = Model(model_input, model_output)
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(), metrics=["accuracy"])

    return model


def all_features_cnn(sequence_length, emb_weights, dis_embeddings_dim, pos_embedding_dim, chunktag_embedding_dim,
                     filter_sizes, nb_filters, num_hidden, dropout_prob, l2_reg):
    """
    Constructs a 1D Convolutional Neural Network utilising sentence, part-of-speech, noun-phrase chunk and
    position features towards sentence classifiation tasks.

    Args:
        sequence_length - Length of the longest sentence.
        emb_weights - Pre-trained embedding weights.
        filter_sizes - Widths of filters (type: list).
        nb_filters - Number of filters for each filter size.
        num_hidden - Number of neurons in hidden layer.
        dropout_prob - Value of dropout to use
        l2_reg - Value of l2 based regularisation.

    Returns:
        Compiled model with Categorical Cross-Entropy loss and Adam optimizer.

    """

    # Word embeddings.
    model_input = Input(shape=(sequence_length,),
                        dtype='int32', name='Sparse_Input_1')
    embedding = Embedding(emb_weights.shape[0], emb_weights.shape[1], input_length=sequence_length,
                          weights=[emb_weights],
                          trainable=True, name='Word_Embeddings')(model_input)

    # Spice distance embedding.
    distance1_input = Input(shape=(sequence_length,),
                            dtype='int32', name='Sparse_Input_D1')
    distance1_embedding = Embedding(62, dis_embeddings_dim, input_length=sequence_length,
                                    trainable=True, name='D1_Embeddings')(distance1_input)

    # Disease distance embedding.
    distance2_input = Input(shape=(sequence_length,),
                            dtype='int32', name='Sparse_Input_D2')
    distance2_embedding = Embedding(62, dis_embeddings_dim, input_length=sequence_length,
                                    trainable=True, name='D2_Embeddings')(distance2_input)

    # POS embeddings.
    pos_input = Input(shape=(sequence_length,),
                      dtype='int32', name='Sparse_Input_POS')
    pos_embedding = Embedding(62, pos_embedding_dim, input_length=sequence_length,
                              trainable=True, name='POS_Embeddings')(pos_input)

    # Chunk Tag embeddings.
    chunktag_input = Input(shape=(sequence_length,),
                           dtype='int32', name='Sparse_Input_ChunkTag')
    chunktag_embedding = Embedding(62, chunktag_embedding_dim, input_length=sequence_length,
                                   trainable=True, name='Chunk_Tag_Embeddings')(chunktag_input)

    # Merge the different embeddings to obtain one embedding for each word.
    embedding = merge([embedding, distance1_embedding, distance2_embedding, pos_embedding, chunktag_embedding],
                      mode='concat',
                      concat_axis=-1)

    conv_blocks = []
    for fs in filter_sizes:
        conv = Convolution1D(filters=nb_filters,
                             kernel_size=fs,
                             padding="valid",
                             activation="relu",
                             strides=1,
                             )(embedding)
        conv = GlobalMaxPool1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dropout(dropout_prob)(z)
    z = Dense(num_hidden, activation="relu", kernel_regularizer=l2(l2_reg))(z)

    model_output = Dense(3, activation="softmax")(z)
    model = Model([model_input, distance1_input, distance2_input,
                   pos_input, chunktag_input], model_output)

    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(), metrics=["accuracy"])
    return model
