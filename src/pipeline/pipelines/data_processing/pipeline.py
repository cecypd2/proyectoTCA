from kedro.pipeline import Pipeline, node, pipeline

from .nodes import limpieza_reservaciones, preprocess_dashboard, create_model_input_table_mean_boost_extra_features, prepare_lstm_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=limpieza_reservaciones,
                inputs="iar_Reservaciones",
                outputs="preprocessed_reservations",
                name="preprocess_reservations_node",
            ),
            node(
                func=preprocess_dashboard,
                inputs=["preprocessed_reservations","iar_canales","iar_estatus_reservaciones","iar_Tipos_Habitaciones"],
                outputs="dashboard_reservations",
                name="preprocess_dashboard_node",
            ),
            node(
                func=create_model_input_table_mean_boost_extra_features,
                inputs="preprocessed_reservations",
                outputs="model_input_table_boost",
                name="create_model_input_table_boost_node",
            ),
            node(
                func=prepare_lstm_data,
                inputs="model_input_table_boost",
                outputs=["X_train","X_test","y_train","y_test","lstm_scaler","lstm_df_feat"],
                name="lstm_data_node",
            )
        ]
    )
