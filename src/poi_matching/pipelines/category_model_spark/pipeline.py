from kedro.pipeline import Pipeline, node, pipeline
from .nodes import  (make_features,
                    train_model,
                    predict,
                    report_accuracy
                    )



def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func=make_features,
                inputs=["df_train", "parameters"],
                outputs="df_features",
                name="make_features",
            ),
            node(
                func=train_model,
                inputs=["df_features", "parameters"],
                outputs="trained_model",
                name="train_model",
            ),
            node(
                func=predict,
                inputs=dict(model='trained_model', df='df_features'),
                outputs="fit_predict_result",
                name="predict",
            ),
            node(
                func=report_accuracy,
                inputs=["fit_predict_result"],
                outputs=None,
                name="report_accuracy",
            ),
            
        ]
    )