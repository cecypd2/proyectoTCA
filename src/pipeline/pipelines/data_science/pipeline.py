from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="modelo",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["modelo","lstm_scaler", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
