import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pca_analysis import (
    clean_data,
    encode_data,
    perform_pca
)

st.set_page_config(
    page_title="PCA Demo",
    layout="wide"
)

st.title(
    "Machine Learning PCA Demo"
)

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.header("Dataset Overview")

    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    st.write(df.isnull().sum())

    df = clean_data(df)

    data = encode_data(df)

    numeric_cols = data.select_dtypes(
        include="number"
    ).columns

    X = data[numeric_cols]

    n_components = st.slider(
        "Number of Components",
        2,
        min(len(numeric_cols), 10),
        2
    )

    pca_df, explained_variance = perform_pca(
        X,
        n_components
    )

    st.header(
        "Reduced Dataset"
    )

    st.dataframe(
        pca_df.head()
    )

    st.header(
        "Explained Variance Ratio"
    )

    st.write(
        explained_variance
    )

    fig, ax = plt.subplots()

    ax.plot(
        range(
            1,
            len(explained_variance) + 1
        ),
        explained_variance,
        marker="o"
    )

    ax.set_xlabel(
        "Principal Components"
    )

    ax.set_ylabel(
        "Explained Variance"
    )

    st.pyplot(fig)

    if n_components >= 2:

        fig2, ax2 = plt.subplots()

        ax2.scatter(
            pca_df.iloc[:, 0],
            pca_df.iloc[:, 1]
        )

        ax2.set_xlabel(
            "PC1"
        )

        ax2.set_ylabel(
            "PC2"
        )

        st.pyplot(fig2)

else:

    st.info(
        "Please Upload a CSV File"
    )