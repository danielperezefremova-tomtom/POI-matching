from kedro.pipeline import Pipeline, node, pipeline
from .nodes import  (transform_to_matching_datamodel,
                    remove_punctuation_and_special_chars_on_names,
                    remove_emoji_patterns)



def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func=transform_to_matching_datamodel,
                inputs=["foursquare_data", "parameters"],
                outputs="foursquare_data_dm",
                name="transform_to_matching_datamodel",
            ),
            node(
                func=remove_punctuation_and_special_chars_on_names,
                inputs=["foursquare_data_dm", "parameters"],
                outputs="foursquare_data_rm_special_chars",
                name="remove_punctuation_and_special_chars_on_names",
            ),
            node(
                func=remove_emoji_patterns,
                inputs=["foursquare_data_rm_special_chars", "parameters"],
                outputs="foursquare_data_rm_emojis",
                name="remove_punctuation_and_special_chars_on_names",
            )
        ]
    )