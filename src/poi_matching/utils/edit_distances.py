import nltk
import numpy as np
import rapidfuzz


def vectorized_haversine(lats1, lats2, longs1, longs2):
    """Calculates haversine distance vectorized.
    Args:
        lats1 (array): Array of latitude 1.
        lats2 (array): Array of longitude 1.
        longs1 (array): Array of latitude 2.
        longs2 (array): Array of longitude 2.
    Returns:
        array: array of haversine distance.
    """
    radius = 6371
    dlat = np.radians(lats2 - lats1)
    dlon = np.radians(longs2 - longs1)
    a = np.sin(
        dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lats1)) * np.cos(
            np.radians(lats2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c
    return d

def check_nan(a, b):
    """
    Checks if both values are not NaN.
    Args:
        a (any type): Any type of value.
        b (any type): Any type of value.
    Returns:
        bool: True if a and b are not NaN, False otherwise.
    """
    if a == a and b == b:
        return True
    else:
        return False


def add_lat_lon_distance_features(df_pairs):
    """Calculates all lat long features for a dataframe with pairs of corrdinates.
    Args:
        df_pairs (dataframe): dataframe with pairs of poi from blocking.
    Returns:
        dataframe: dataframe with lat long features.
    """
    lat1 = df_pairs["latitude1"]
    lat2 = df_pairs["latitude2"]
    lon1 = df_pairs["longitude1"]
    lon2 = df_pairs["longitude2"]
    df_pairs["haversine"] = vectorized_haversine(lat1, lat2, lon1, lon2)
    col_64 = list(df_pairs.dtypes[df_pairs.dtypes == np.float64].index)
    for col in col_64:
        df_pairs[col] = df_pairs[col].astype(np.float32)
    return df_pairs


def fast_cosine(vec1, vec2, batch=10000, is_batch=True):
    """Function to calculate cosine similarity faster in vectorized form.
    Args:
        vec1 (array): array of embeddings1.
        vec2 (array): array of embeddings2.
        batch (int, optional): batch size of vectors to calculate cosine similarity. Defaults to 1000000.
        is_batch (bool, optional): To batch or not. Defaults to True.
    Returns:
        array: array of cosine similarity.
    """

    if is_batch:

        sims = np.empty((0), np.float32)

        for i in range(0, len(vec1), batch):
            cosine = np.sum(vec1[i:i + batch] * vec2[i:i + batch], axis=1)
            cosine = np.round(cosine, 3)
            sims = np.concatenate((sims, cosine))
        return sims
    else:
        cosine = np.sum(vec1 * vec2, axis=1)
        cosine = np.round(cosine, 3)
    return cosine


def strike_a_match(str1, str2):
    """Dice bigram calculation.
    Args:
        str1 (text): text1
        str2 (text): text2
    Returns:
        float: dice bigram score
    """
    if check_nan(str1, str2):
        pairs1 = set(nltk.bigrams(str1))
        pairs2 = set(nltk.bigrams(str2))
        hit_count = len(set(pairs1) & set(pairs2))
        union = len(pairs1) + len(pairs2)
        try:
            return (2.0 * hit_count) / union
        except:
            if str1 == str2:
                return 1.0
            else:
                return 0.0
    else:
        return -1


def sorted_winkler(str1, str2):
    """find edit jaro wrinkler distance after sorting both the strings.
    Args:
        str1 (text): text1
        str2 (text): text2
    Returns:
        float: jarowinkler similarity of sorted strings.
    """
    if check_nan(str1, str2):
        a = " ".join(sorted(str1.split()))
        b = " ".join(sorted(str2.split()))
        return rapidfuzz.distance.JaroWinkler.similarity(a, b)
    else:
        return -1


def davies(str1, str2):
    """https://www.tandfonline.com/doi/full/10.1080/17538947.2017.1371253
    Args:
        str1 (text): text1
        str2 (text): text2
    Returns:
        float: score
    """
    if check_nan(str1, str2):
        a = str1.split()
        b = str2.split()

        a = frozenset(a)
        b = frozenset(b)
        aux1 = sorted_winkler(str1, str2)
        jw_scores = {(i, j): rapidfuzz.distance.JaroWinkler.similarity(i, j) for i in a for j in b}
        intersection_length = (sum(max(jw_scores[i, j] for j in b) for i in a) + sum(max(jw_scores[i, j] for i in a) for j in b))/2
        aux2 = intersection_length / (len(a) + len(b) - intersection_length)
        return (aux1 + aux2) / 2
    else:
        return -1

def lcs(text1, text2):
    """ function two find lcs similarity between two text.
    Args:
        text1 (str): text 1
        text2 (str): text 2
    Returns:
        float: lcs score
    """
    if check_nan(text1, text2):
        return rapidfuzz.distance.LCSseq.similarity(
            str(text1), str(text2)) / min(len(text1), len(text2))
    else:
        return -1


def jaro(text1, text2):
    """ function two find jaro similarity between two text.
    Args:
        text1 (str): text 1
        text2 (str): text 2
    Returns:
        float: jaro score
    """
    if check_nan(text1, text2):
        return rapidfuzz.distance.Jaro.similarity(text1, text2)
    else:
        return -1


def leven(text1, text2):
    """ function two find Damerau Levenshtein similarity between two text.
    Args:
        text1 (str): text 1
        text2 (str): text 2
    Returns:
        float: Damerau Levenshtein score
    """
    if check_nan(text1, text2):
        return rapidfuzz.distance.DamerauLevenshtein.normalized_similarity(
            text1, text2)
    else:
        return -1


def token_set_ratio(text1, text2):
    """ function two find token set ratio  between two text.
    Args:
        text1 (str): text 1
        text2 (str): text 2
    Returns:
        float: token set ratio score
    """
    if check_nan(text1, text2):
        return rapidfuzz.token_set_ratio(text1, text2) / 100
    else:
        return -1


def WRatio(text1, text2):
    """ function two find word ratio between two text.
    Args:
        text1 (str): text 1
        text2 (str): text 2
    Returns:
        float: word ratio score
    """
    if check_nan(text1, text2):
        return rapidfuzz.WRatio(text1, text2) / 100
    else:
        return -1


def ratio(text1, text2):
    """ function two find ratio between two text.
    Args:
        text1 (str): text 1
        text2 (str): text 2
    Returns:
        float: ratio score
    """
    if check_nan(text1, text2):
        return rapidfuzz.ratio(text1, text2) / 100
    else:
        return -1


def QRatio(text1, text2):
    """ function two find QRatio between two text.
    Args:
        text1 (str): text 1
        text2 (str): text 2
    Returns:
        float: QRatio score
    """
    if check_nan(text1, text2):
        return rapidfuzz.QRatio(text1, text2) / 100
    else:
        return -1
