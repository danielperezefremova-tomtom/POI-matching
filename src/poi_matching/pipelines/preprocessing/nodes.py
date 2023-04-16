from poi_matching.utils.geohash import encode_location
from poi_matching.utils.similarity import SimilarityModel
import pandas as pd
import numpy as np
import logging 
from pyspark.sql.functions import col, lower, lit, split, array_join, translate
import pyspark 
import advertools as adv
from functools import reduce
from pyspark.ml.feature import StopWordsRemover, Tokenizer


SPECIAL_CHARS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~'

log = logging.getLogger(__name__)

def make_features(df:pd.DataFrame, parameters: dict)->pd.DataFrame:

    similarity_model = SimilarityModel(model_name=parameters['embedding_model'], batch=32)
    df['match'] = df['match'].astype(int)

    locations = np.array(list(zip(df['ref_lat'], df['ref_lon'])))
    locations_encoded = np.array([encode_location(*loc) for loc in locations])
    df.loc[:, 'ref_lat_encoded'] = locations_encoded[:, 0].tolist()
    df.loc[:, 'ref_lon_encoded'] = locations_encoded[:, 1].tolist()

    log.info(f'Geohashed {locations.shape[0]} locations')

    names_encoded = np.array([similarity_model.encode_string(name) for name in df['name']])
    log.info(f'Encoded {names_encoded.shape[0]} Names')
    df['names_encoded'] = names_encoded.tolist()

    cat_encoded = np.array([similarity_model.encode_string(name) for name in df['category_name']])
    log.info(f'Encoded {cat_encoded.shape[0]} Category names')
    df['categories_encoded'] = cat_encoded.tolist()

    columns_to_keep = ['matching_run_id',
                        'reference_id', 
                        'match', 
                        'ref_lat_encoded',
                        'ref_lon_encoded',
                        'names_encoded',
                        'categories_encoded',
                        'category_name'
                        ]
    

    return df[columns_to_keep]

def transform_to_matching_datamodel(df: pyspark.sql.DataFrame,
                                     parameters: dict) -> pyspark.sql.DataFrame:
    
    for column, new_column in parameters['columns_map'].items():

        _name, _type = new_column['name'], new_column['type']
        df = df.withColumnRenamed(column, _name) \
                .withColumn(_name, col(_name).cast(_type))

    df = df.withColumn('run_id', lit(parameters['run_id']))
    return df

def remove_punctuation_and_special_chars_on_names(df: pyspark.sql.DataFrame,
                                        parameters: dict) -> pyspark.sql.DataFrame:

    for column in ['name_1', 'name_2']:
        df=df.withColumn(column, translate(column, SPECIAL_CHARS, ''))

    return df

def transform_array_to_string(df: pyspark.sql.DataFrame,
                            input_columns:list,
                            output_columns:list,
                            pattern:str) -> pyspark.sql.DataFrame:

    columns_list = list(zip(input_columns, output_columns))
    for column, new_column in columns_list:
        df = df.withColumn(new_column, array_join(col(column), pattern))

    return df

def generate_bag_of_stopwords(list_of_languages: list) -> list:
    
    stopwords_dictionary = {language: adv.stopwords[language] for language in list_of_languages}
    bag_of_stopwords = list(reduce(lambda x,y: x | y, stopwords_dictionary.values()))

    return bag_of_stopwords

def remove_stopwords(df: pyspark.sql.DataFrame,
                    input_columns:list,
                    output_columns:list,
                    stopwords: list) -> pyspark.sql.DataFrame:

    remover = StopWordsRemover(stopWords=stopwords)
    remover.setInputCols(input_columns)
    remover.setOutputCols(output_columns)

    df_transformed = remover.transform(df)

    return df_transformed

def tokenize(df: pyspark.sql.DataFrame,
                    input_columns:list,
                    output_columns:list) -> pyspark.sql.DataFrame:
    
    columns_list = list(zip(input_columns, output_columns))
    tokenizer = Tokenizer()

    for in_col, out_col in columns_list:
        tokenizer.setInputCol(in_col)
        tokenizer.setOutputCol(out_col)
        df = tokenizer.transform(df)

    return df
    
def remove_stopwords_on_names(df: pyspark.sql.DataFrame,
                     parameters: dict) -> pyspark.sql.DataFrame:

    df = tokenize(df, input_columns=['name_1', 'name_2'], output_columns=['name_1_tokenized', 'name_2_tokenized'])

    stopwords = generate_bag_of_stopwords(parameters['stopwords']['languages'])
    enriched_stopwords = stopwords + parameters['stopwords']['additional_stopwords']

    df_transformed = remove_stopwords(df,
                                    input_columns=['name_1_tokenized', 'name_2_tokenized'],
                                    output_columns=['name_1_stopwords_filtered', 'name_2_stopwords_filtered'],
                                    stopwords=enriched_stopwords)
    
    df_processed = transform_array_to_string(df_transformed,
                                            input_columns=['name_1_stopwords_filtered', 'name_2_stopwords_filtered'],
                                            output_columns=['name_1_filtered', 'name_2_filtered'],
                                            pattern=' ')
    return df_processed






