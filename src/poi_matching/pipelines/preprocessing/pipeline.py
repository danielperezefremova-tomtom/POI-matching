from kedro.pipeline import Pipeline, node, pipeline
from .nodes import  (transform_to_matching_datamodel,
                    remove_punctuation_and_special_chars_on_names,
                    remove_emoji_patterns,
                    explode_categories,
                    select_columns,
                    filter_by_country
                    )



def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func=transform_to_matching_datamodel,
                inputs=["df_input", "parameters"],
                outputs="df_transformed_to_data_model",
                name="transform_to_matching_datamodel",
            ),
            node(
                func=filter_by_country,
                inputs=["df_transformed_to_data_model", "parameters"],
                outputs="df_filtered",
                name="filter_by_country",
            ),
            node(
                func=remove_punctuation_and_special_chars_on_names,
                inputs=["df_filtered", "parameters"],
                outputs="df_rm_special_chars",
                name="remove_punctuation_and_special_chars_on_names",
            ),
            node(
                func=remove_emoji_patterns,
                inputs=["df_rm_special_chars", "parameters"],
                outputs="df_rm_emojis",
                name="remove_emoji_patterns",
            ),
            node(
                func=explode_categories,
                inputs=["df_rm_emojis", "parameters"],
                outputs="df_explode_categories",
                name="explode_categories",
            ),
            node(
                func=select_columns,
                inputs=["df_explode_categories", "parameters"],
                outputs="df_preprocessed",
                name="select_columns",
            )
        ]
    )