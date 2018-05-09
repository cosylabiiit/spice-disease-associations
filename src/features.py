import numpy as np
from geniatagger import GeniaTagger


def extract_chunk_features(tagger_out):
    """ Return the noun-phrase chunk tag for each token. """

    return [w[3] for w in tagger_out[1]]


def extract_pos_features(tagger_out):
    """ Return the part-of-speech tag for each token. """

    return [w[2] for w in tagger_out[1]]


def extract_word_tokens(tagger_out):
    """ Returns tokenized sentence. """

    return [w[0] for w in tagger_out[1]]


def extract_distance_features(tagger, tagger_out, food_text, dis_text, maximum_distance):
    """ Returns the distance of each token from food-token and disease-token. """

    entsen_genia_out, origsen_genia_out = tagger_out
    max_d, min_d = maximum_distance, -maximum_distance

    # Tokenize food mention and disease mention.
    tok_food_text = tagger.parse(food_text)
    tok_dis_text = tagger.parse(dis_text)
    len_f = len(tok_food_text)
    len_d = len(tok_dis_text)

    # Get initial id of food and disease.
    word2id = {w[0]: i for i, w in enumerate(entsen_genia_out)}
    f_st = word2id['Foodentity']
    d_st = word2id['Diseaseentity']

    # Get distance of tokens from food-entity and disease-entity
    tokens = [p[0] for p in origsen_genia_out]

    if f_st < d_st:
        if len_f > 1:
            d_st += len_f - 1
    else:
        if len_d > 1:
            f_st += len_d - 1

    f_end = f_st + len_f
    d_end = d_st + len_d

    food_distance = np.zeros((len(tokens)))
    food_distance[:f_st] = np.arange(start=-f_st, stop=0, step=1)
    food_distance[f_end:] = np.arange(
        start=f_end + 1, stop=len(tokens) + 1, step=1) - f_end
    food_distance[food_distance > max_d] = max_d
    food_distance[food_distance < min_d] = min_d

    dis_distance = np.zeros((len(tokens)))
    dis_distance[:d_st] = np.arange(start=-d_st, stop=0, step=1)
    dis_distance[d_end:] = (
        np.arange(start=d_end + 1, stop=len(tokens) + 1, step=1) - d_end)
    dis_distance[dis_distance > max_d] = max_d
    dis_distance[dis_distance < min_d] = min_d

    return food_distance, dis_distance


def extract_features(df, genia_loc, max_d):
    """
    Extracts part-of-speech, noun-phrase chunk and position features from a sentence.

    Args:
        df - Pandas DataFrame containing sentence and tagged food-entity and disease entity.
        genia_loc - Path to geniatagger executable.
        max_d - Value for maximum distance.

    Returns:
        Tuple containing tokenized sentences and, position, noun-phrase chunk
        and part-of-speech features.
    """

    fd, dd, chunkf, posf, wtokens = [], [], [], [], []
    for i, row in df.iterrows():

        # Replace entity markers with original text.
        sen = row['Preprocessed']
        orig_sen = sen.replace('[F]', row['Food Text'])
        orig_sen = orig_sen.replace('[D]', row['Disease Text'])

        # Replace entity markers with tagger compatible markers.
        ent_sen = sen.replace('[F]', 'Foodentity')
        ent_sen = ent_sen.replace('[D]', 'Diseaseentity')

        # Initialize Genia Tagger and parse sentences.
        tagger = GeniaTagger(genia_loc)
        tagger_out = tagger.parse(orig_sen), tagger.parse(ent_sen)

        # Tokenize sentence to workds
        word_token = extract_word_tokens(tagger_out)
        wtokens.append(word_token)

        # Get position features.
        distances = extract_distance_features(
            tagger, tagger_out, row['Food Text'], row['Disease Text'], max_d)
        fd.append(distances[0])
        dd.append(distances[1])

        # Get chunk tag features.
        chunk_tags = extract_chunk_features(tagger_out)
        chunkf.append(chunk_tags)

        # Get part-of-speech tags for each token.
        pos_tags = extract_pos_features(tagger_out)
        posf.append(pos_tags)

    return wtokens, fd, dd, chunkf, posf
