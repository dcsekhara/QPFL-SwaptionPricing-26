from .factory import build_model_from_config


SUPPORTED_MODELS = (
    "linear",
    "ridge",
    "mlp",
    "xgboost",
    "cnn",
    "lstm",
    "mlp_torch",
)


def create_model(model_config):
    return build_model_from_config(model_config)
