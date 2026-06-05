import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.decomposition import PCA


def clean_data(df):

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(
            df[col].mean()
        )

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    return df.drop_duplicates()


def encode_data(df):

    data = df.copy()

    for col in data.columns:

        if not pd.api.types.is_numeric_dtype(
            data[col]
        ):

            le = LabelEncoder()

            data[col] = le.fit_transform(
                data[col].astype(str)
            )

    return data


def perform_pca(
    X,
    n_components
):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=n_components
    )

    transformed = pca.fit_transform(
        X_scaled
    )

    columns = [
        f"PC{i+1}"
        for i in range(n_components)
    ]

    pca_df = pd.DataFrame(
        transformed,
        columns=columns
    )

    return (
        pca_df,
        pca.explained_variance_ratio_
    )