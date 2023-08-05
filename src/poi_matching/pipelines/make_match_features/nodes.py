import numpy as np
from rapidfuzz import fuzz
from pyspark.sql.functions import (udf, col, lower)
import phonetics
import difflib
import pylcs
import Levenshtein
import phonetics
from haversine import haversine, Unit
import pyspark
from poi_matching.utils.similarity import SimilarityModel
from pyspark.sql.types import FloatType, ArrayType, StringType
import logging 
import torch
import jaro

TRANSFORMER_MODEL = SimilarityModel(batch=128)
log = logging.getLogger(__name__)

def fuzzy_matching(text: str, target: str) -> float:
    """fuzzy string matching between a candidate and a target name
  :param text: name to compare with target
  :type text: str
  :param target: target name
  :type target: str
  :return: fuzzy matching score
  :rtype: float
  """
    matching_score = 0

    if text is None:
        return matching_score

    matching_score = fuzz.token_set_ratio(target,text)

    return matching_score/100

udf_fuzzy_token_ratio = udf(fuzzy_matching)

def fuzzy_matching_ratio(text: str, target: str) -> float:
    """
  """
    matching_score = 0

    if text is None:
        return matching_score

    matching_score = fuzz.ratio(target,text)

    return matching_score/100

udf_fuzzy_ratio = udf(fuzzy_matching_ratio)

def fuzzy_matching_token_sort_ratio(text: str, target: str) -> float:
    """
  """
    matching_score = 0

    if text is None:
        return matching_score

    matching_score = fuzz.token_sort_ratio(target,text)

    return matching_score/100

udf_token_sort_ratio = udf(fuzzy_matching_token_sort_ratio)

def fuzzy_matching_partial_ratio(text: str, target: str) -> float:
    """
  """
    matching_score = 0

    if text is None:
        return matching_score

    matching_score = fuzz.partial_ratio(target,text)

    return matching_score/100

udf_partial_ratio = udf(fuzzy_matching_partial_ratio)

def gestalt_matching(text: str, target: str) -> float:
    """
  """
    s = difflib.SequenceMatcher(None, text, target)
    return s.ratio()

udf_gestalt_ratio = udf(gestalt_matching)

def lcs_ratio(text: str, target: str) -> float:
    max_common_len = pylcs.lcs_string_length(text, target)
    min_str_len = min(len(text), len(target))
    
    return max_common_len/min_str_len

udf_lcs_ratio = udf(lcs_ratio)


def levenshtein_ratio(text: str, target: str) -> float:
    
    edits = Levenshtein.distance(text, target)
    max_len_str = max(len(text), len(target))
    
    if max_len_str>0:
        return 1-edits/max_len_str
    else:
        return 0
    
def metaphone_ratio(text: str, target: str) -> float:
    
    text_phonetic = phonetics.dmetaphone(text)[0]
    target_phonetic = phonetics.dmetaphone(target)[0]
    if isinstance(text_phonetic, str) and isinstance(target_phonetic, str):
    
        return levenshtein_ratio(text_phonetic, target_phonetic)
    
    else:
        return None

udf_metaphone_ratio = udf(metaphone_ratio)

def haversine_meters(lat1, lon1, lat2, lon2):
    
    return haversine((lat1, lon1), (lat2,lon2), unit=Unit.METERS)

udf_haversine = udf(haversine_meters)

def ngrams(sequence, n):
    """Create ngrams from sequence, e.g. ([1,2,3], 2) -> [(1,2), (2,3)]
       Note that fewer sequence items than n results in an empty list being returned"""
    # credit: http://stackoverflow.com/questions/2380394/simple-implementation-of-n-gram-tf-idf-and-cosine-similarity-in-python
    sequence = list(sequence)
    count = max(0, len(sequence) - n + 1)
    return [''.join(tuple(sequence[i:i+n])) for i in range(count)]

def ngrams_ratio(string1, string2, int_grams=3):

    string1_ngrams = set(ngrams(sequence=string1, n=int_grams))
    string2_ngrams = set(ngrams(sequence=string2, n=int_grams))
    if (len(string1_ngrams)==0) or (len(string1_ngrams)==0):
        return 0
    intersection = string1_ngrams.intersection(string2_ngrams)
    intersection_len = len(intersection)

    return 2*intersection_len / (len(string1_ngrams) + len(string2_ngrams))

udf_trigrams = udf(ngrams_ratio)

def jaro_winkler(string1, string2):
    result = jaro.jaro_winkler_metric(string1, string2)
    return result

udf_jaro_winkler = udf(jaro_winkler)


def make_name_features(df: pyspark.sql.DataFrame,
                     parameters: dict) -> pyspark.sql.DataFrame:
    
    df_transformed = df.withColumn('name_1', lower(col('name_1'))) \
                        .withColumn('name_2', lower(col('name_2'))) \
                        .withColumn('names_token_ratio', udf_fuzzy_token_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('names_ratio', udf_fuzzy_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('names_token_sort_ratio', udf_token_sort_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('names_partial_ratio', udf_partial_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('gestalt_ratio', udf_gestalt_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('lcs_ratio', udf_lcs_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('metaphone_ratio', udf_metaphone_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('jaro_winkler', udf_jaro_winkler(col('name_1'), col('name_2'))) \
                        .withColumn('trigrams_ratio', udf_trigrams(col('name_1'), col('name_2'))) \

    return df_transformed

def make_location_features(df: pyspark.sql.DataFrame,
                     parameters: dict) -> pyspark.sql.DataFrame:
    
    df_transformed = df.withColumn('distance', udf_haversine(col('latitude_1'), col('longitude_1'), 
                                                            col('latitude_2'), col('longitude_2')))

    return df_transformed

def make_category_features(df: pyspark.sql.DataFrame, category_model,
                     parameters: dict) -> pyspark.sql.DataFrame:
    log.info(category_model)
    udf_compute_encoding = udf(lambda x: TRANSFORMER_MODEL.encode_string(x).tolist(), ArrayType(FloatType()))

    df_encoded=df.withColumn('category_1', lower(col('category_1'))) \
                    .withColumn('category_2', lower(col('category_2'))) \
                    .withColumn('category_1_embeding', udf_compute_encoding(col('category_1'))) \
                    .withColumn('category_2_embeding', udf_compute_encoding(col('category_2')))    

    return df_encoded


@udf(FloatType())
def udf_cosine_sim(vector1, vector2):  # input: pd.Series; output: pd.Series
    
    cos_sim_instance = torch.nn.CosineSimilarity(dim=0)
    tensor1 = torch.Tensor(vector1)
    tensor2 = torch.Tensor(vector2)
    
    return round(float(cos_sim_instance(tensor1, tensor2)),2)

def compute_category_similarity(df: pyspark.sql.DataFrame,
                     parameters: dict) -> pyspark.sql.DataFrame:
    
    df_transformed = df.withColumn('category_semantic_similarity', udf_cosine_sim(col('category_1_embeding'),
                                                                                col('category_2_embeding')))
    return df_transformed

