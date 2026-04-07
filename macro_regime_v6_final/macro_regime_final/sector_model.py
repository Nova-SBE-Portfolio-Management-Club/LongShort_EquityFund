import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class SectorReturnModel:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.models = {}
        self.scalers = {}

    def fit(self, X_q: pd.DataFrame, Y_q: pd.DataFrame):
        """
        X_q: columns MultiIndex (Sector, Feature)
        Y_q: columns = sectors, values = next quarter returns
        For each sector: train on its own features only.
        """
        for sector in Y_q.columns:
            Xi = X_q[sector].dropna()
            yi = Y_q[sector].dropna()
            common = Xi.index.intersection(yi.index)
            Xi = Xi.loc[common]
            yi = yi.loc[common]

            if len(common) < 20:
                continue

            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xi.values)
            model = Ridge(alpha=self.alpha)
            model.fit(Xs, yi.values)

            self.models[sector] = model
            self.scalers[sector] = scaler

    def predict_one(self, X_q_last: pd.DataFrame) -> dict:
        """
        Predict for one quarter using latest feature row.
        X_q_last: a single-row DataFrame with MultiIndex columns.
        """
        preds = {}
        for sector, model in self.models.items():
            if sector not in X_q_last.columns.get_level_values(0):
                continue
            Xi = X_q_last[sector].copy()
            if Xi.isna().any().any():
                continue
            scaler = self.scalers[sector]
            Xs = scaler.transform(Xi.values)
            preds[sector] = float(model.predict(Xs)[0])
        return preds
