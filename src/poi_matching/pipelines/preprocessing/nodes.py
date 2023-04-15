from poi_matching.utils.geohash import encode_location
from poi_matching.utils.similarity import SimilarityModel
import pandas as pd
import numpy as np
import logging 
from pyspark.sql.functions import *

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

def transform_to_matching_datamodel(df, parameters):
    
    for column, new_column in parameters['columns_map'].items():
        df = df.withColumnRenamed(column, new_column['name']) \
                .withColumn(new_column['name'], col(new_column['name']).cast(new_column['type']))

    df = df.withColumn('run_id', lit(parameters['run_id']))
    return df
