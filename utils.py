import numpy as np


class utils:
    def __init__(self, in_dir: str):
        self.in_dir = in_dir

    @staticmethod
    def load_data(path: str) -> np.ndarray:
        return np.load(path, allow_pickle=True)

    @staticmethod
    def norm_image(x: np.ndarray) -> np.ndarray:
        return x / 255.0

    @staticmethod
    def denorm_image(x: np.ndarray) -> np.ndarray:
        return x * 255.0

    @staticmethod
    def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
        return np.eye(num_classes)[y]

