from kedro.pipeline import Pipeline, node, pipeline
from .nodes import make_features, \
                    transform_to_matching_datamodel



def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func=transform_to_matching_datamodel,
                inputs=["foursquare_data", "parameters"],
                outputs="foursquare_data_dm",
                name="transform_to_matching_datamodel",
            ),
        ]
    )