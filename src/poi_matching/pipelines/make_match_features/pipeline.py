from kedro.pipeline import Pipeline, node, pipeline
from .nodes import  (make_name_features,
                     make_location_features,
                    )


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func=make_name_features,
                inputs=["df_preprocessed", "parameters"],
                outputs="df_name_features",
                name="make_name_features",
            ),
            node(
                func=make_location_features,
                inputs=["df_name_features", "parameters"],
                outputs="df_location_features",
                name="make_location_features",
            ),
       ]
    )