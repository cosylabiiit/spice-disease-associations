import re
from keras.preprocessing.sequence import pad_sequences


def remove_bracket_words(sentence):
    """ Removes words inside brackets if they don't contain a food
    or a disease term. """

    repl = re.findall(r'[\(\[].*?[\)\]]', sentence)
    for r in repl:
        if 'ANN-' not in r:
            sentence = sentence.replace(r, '')

    # Remove extra whitespace.
    sentence = re.sub(' +', ' ', sentence)

    return sentence


def clean_text(sentence):
    """ Preprocesses input sentence. """

    sentence = remove_bracket_words(sentence)
    sentence = re.sub(r"``.*?:", ':', sentence)

    sentence = sentence.replace(r"ANN-FOOD", ' [F] ')
    sentence = sentence.replace(r"ANN-DISEASE", ' [D] ')

    sentence = re.sub(r"[^A-Za-z0-9(),!?;:\[\]\.\%]", " ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)

    words = []
    for word in sentence.split():
        word = word.strip()
        if re.search(r'[\d]', word):
            if not re.search(r'[A-Za-z]', word):
                word = re.sub(r'[\d]{1,}', '1', word)

        words.append(word)

    return ' '.join(words)


def model_preprocess(ann_features, unann_features, max_d):
    """ Preprocesses the extracted features and gets them in the expected format for the ConvNet.

    Args:
        ann_features - Output of extract_features function on annotated text.
        unann_features - Output of extract_features function on unannotated text.

    Returns:
        ann_features - Padded and numericalized features.
        unann_features - Padded and numericalized features.
        id2wordtoken - Mapping from numerical ids to word strings.
        MAX_LEN - Length of the longest sequence.

    """

    # 1. Zero Pad
    # Find maximum length
    MAX_LEN = max(ann_features[-1], unann_features[-1])

    ann_features[0] = pad_sequences(
        ann_features[0], value=max_d + 1, maxlen=MAX_LEN, padding="post")
    ann_features[1] = pad_sequences(
        ann_features[1], value=max_d + 1, maxlen=MAX_LEN, padding="post")
    unann_features[0] = pad_sequences(
        unann_features[0], value=max_d + 1, maxlen=MAX_LEN, padding="post")
    unann_features[1] = pad_sequences(
        unann_features[1], value=max_d + 1, maxlen=MAX_LEN, padding="post")

    # 2. Chunk tags to integers and pag.
    # Exhaustive set of chunk tags
    chunk_tags = set([c for chunktag in ann_features[2] for c in chunktag] +
                     [c for chunktag in unann_features[2] for c in chunktag])

    # Mapping from chunk tag to id and vice-versa.
    chunktag2id = {chunktag: i for i, chunktag in enumerate(chunk_tags)}
    id2chunktag = {i: chunktag for chunktag, i in chunktag2id.items()}

    # Replace chunktags by numbers.
    ann_features[2] = [[chunktag2id[c] for c in chunk_features]
                       for chunk_features in ann_features[2]]
    unann_features[2] = [[chunktag2id[c] for c in chunk_features]
                         for chunk_features in unann_features[2]]

    # Pad the chunk tag sequences.
    ann_features[2] = pad_sequences(ann_features[2], value=len(
        id2chunktag), maxlen=MAX_LEN, padding="post")
    unann_features[2] = pad_sequences(unann_features[2], value=len(
        id2chunktag), maxlen=MAX_LEN, padding="post")

    # 3. Part-of-speech tags to integers and pad
    # Exhaustive set of POS tags
    pos_tags = set([c for postag in ann_features[3] for c in postag] +
                   [c for postag in unann_features[3] for c in postag])

    # Mapping from pos tag to id and vice-versa.
    postag2id = {postag: i for i, postag in enumerate(pos_tags)}
    id2postag = {i: postag for postag, i in postag2id.items()}

    # Replace postags by numbers.
    ann_features[3] = [[postag2id[c] for c in pos_features]
                       for pos_features in ann_features[3]]
    unann_features[3] = [[postag2id[c] for c in pos_features]
                         for pos_features in unann_features[3]]

    # Pad the pos tag sequences.
    ann_features[3] = pad_sequences(ann_features[3], value=len(
        id2postag), maxlen=MAX_LEN, padding="post")
    unann_features[3] = pad_sequences(unann_features[3], value=len(
        id2postag), maxlen=MAX_LEN, padding="post")

    # 4. Numericalize word tokens.
    word_tokens = set([c for wordtoken in ann_features[4] for c in wordtoken] +
                      [c for wordtoken in unann_features[4] for c in wordtoken])

    # Mapping from word token to id and vice-versa.
    wordtoken2id = {wordtoken: i for i, wordtoken in enumerate(word_tokens)}
    id2wordtoken = {i: wordtoken for postag, i in wordtoken2id.items()}

    # Replace word tokens by numbers.
    ann_features[4] = [[wordtoken2id[c] for c in wordtokens]
                       for wordtokens in ann_features[4]]
    unann_features[4] = [[wordtoken2id[c] for c in wordtokens]
                         for wordtokens in unann_features[4]]

    # Pad
    ann_features[4] = pad_sequences(ann_features[4], value=len(
        id2wordtoken), maxlen=MAX_LEN, padding="post")
    unann_features[4] = pad_sequences(unann_features[4], value=len(
        id2wordtoken), maxlen=MAX_LEN, padding="post")

    return ann_features, unann_features, id2wordtoken, MAX_LEN
