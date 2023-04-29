from poi_matching.utils.similarity import SimilarityModel
from pyspark.sql.types import FloatType, ArrayType
from pyspark.ml.functions import array_to_vector
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import logging 
from pyspark.sql.functions import (col,
                                    udf,
                                    )
import pyspark 


def make_features(df: pyspark.sql.DataFrame, parameters: dict
                  ) -> pyspark.sql.DataFrame:
    

    model_name = parameters['cat_model']['embeding']['model_name']
    batch = parameters['cat_model']['embeding']['batch_size']
    input_col = parameters['cat_model']['input_col']
    label_col = parameters['cat_model']['label_col']

    model = SimilarityModel(model_name=model_name, 
                            batch=batch)
    
    udf_compute_encoding = udf(lambda x: model.encode_string(x).tolist(), ArrayType(FloatType()))

    df_encoded = df.withColumn(f'{input_col}_encoded', udf_compute_encoding(col(input_col))) \
                    .withColumn('features', array_to_vector(col(f'{input_col}_encoded')))

    df_transformed = df_encoded.select('features', label_col)

    return df_transformed


def train_model(df: pyspark.sql.DataFrame, parameters: dict
                ) -> pyspark.sql.DataFrame:
    
    label_col = parameters['cat_model']['label_col']
    clf_params = parameters['cat_model']['clf_params']

    label_encoder = StringIndexer(inputCol = label_col, outputCol = 'label', handleInvalid='error')
    clf = RandomForestClassifier(**clf_params)
    pipeline = Pipeline(stages=[label_encoder, clf])

    # df = df.repartition(100)
    model = pipeline.fit(df)

    return model

def predict(df: pyspark.sql.DataFrame, model: pyspark.ml.Pipeline
                ) -> pyspark.sql.DataFrame:
    """Node for making predictions given a pre-trained model and a testing dataset.
    """
    prediction = model.transform(df)
    return prediction

def report_accuracy(predictions: pyspark.sql.DataFrame) -> None:
    """Node for reporting the accuracy of the fit-predict performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    log = logging.getLogger(__name__)
    log.info("Model accuracy: %0.2f%%", accuracy * 100)


