import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreprocessor:
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            categories=[
                [
                    "Majority low-income zone",
                    "Approximately 50% low-income zone",
                    "Minority low-income zone",
                    "Not low-income zone",
                ]
            ],
        )

        self.norm_constants = {}  # Store means and stds per continuous feature
        self.one_hot_columns = []

    def fit(self, df: pd.DataFrame):
        # Fit scaler
        cont_data = df[self.continuous_features]
        self.scaler.fit(cont_data)
        self.norm_constants = {
            col: {"mean": m, "std": s}
            for col, m, s in zip(
                cont_data.columns, self.scaler.mean_, np.sqrt(self.scaler.var_)
            )
        }

        # Fit encoder
        cat_data = df[self.categorical_features]
        self.encoder.fit(cat_data)
        self.one_hot_columns = self.encoder.get_feature_names_out(
            self.categorical_features
        ).tolist()

    def transform(self, df: pd.DataFrame):
        cont_data = self.scaler.transform(df[self.continuous_features])
        cat_data = self.encoder.transform(df[self.categorical_features])
        features = np.concatenate([cont_data, cat_data], axis=1)
        return features

    def save_constants(self, filepath):
        import json

        with open(filepath, "w") as f:
            json.dump(
                {
                    "norm_constants": self.norm_constants,
                    "one_hot_columns": self.one_hot_columns,
                },
                f,
            )

    def load_constants(self, filepath):
        import json

        with open(filepath, "r") as f:
            d = json.load(f)
        self.norm_constants = d["norm_constants"]
        self.one_hot_columns = d["one_hot_columns"]
