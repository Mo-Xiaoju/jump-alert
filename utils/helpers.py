import os


def list_model_files(models_dir: str):
    if not os.path.isdir(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith(".onnx")]
