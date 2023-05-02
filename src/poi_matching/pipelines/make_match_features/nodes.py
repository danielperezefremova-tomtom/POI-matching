import numpy as np
from rapidfuzz import fuzz
from pyspark.sql.functions import (udf, col, levenshtein)
import phonetics
import seaborn as sns
import difflib
import pylcs
import Levenshtein
import phonetics
from haversine import haversine, Unit
import pyspark
from poi_matching.utils.similarity import SimilarityModel
from pyspark.sql.types import FloatType, ArrayType, StringType
import logging 

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

def make_name_features(df: pyspark.sql.DataFrame,
                     parameters: dict) -> pyspark.sql.DataFrame:
    
    df_transformed = df.withColumn('names_token_ratio', udf_fuzzy_token_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('names_ratio', udf_fuzzy_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('names_token_sort_ratio', udf_token_sort_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('names_partial_ratio', udf_partial_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('gestalt_ratio', udf_gestalt_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('lcs_ratio', udf_lcs_ratio(col('name_1'), col('name_2'))) \
                        .withColumn('metaphone_ratio', udf_metaphone_ratio(col('name_1'), col('name_2')))
    
    return df_transformed

def make_location_features(df: pyspark.sql.DataFrame,
                     parameters: dict) -> pyspark.sql.DataFrame:
    
    df_transformed = df.withColumn('distance', udf_haversine(col('latitude_1'), col('longitude_1'), 
                                                            col('latitude_2'), col('longitude_2')))

    return df_transformed

def make_category_features(df: pyspark.sql.DataFrame, category_model,
                     parameters: dict) -> pyspark.sql.DataFrame:
    log.info(category_model)

    predict_category_udf = udf(lambda x: str(category_model.predict(TRANSFORMER_MODEL.encode_string(x).reshape(1, -1))[0]))

    df_transformed=df.withColumn('category_1_projection', predict_category_udf(col('category_1'))) \
                    .withColumn('category_2_projection', predict_category_udf(col('category_2')))
    
    return df_transformed