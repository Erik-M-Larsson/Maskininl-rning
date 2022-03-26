import pandas as pd
import joblib

df_test_samples = pd.read_csv("Labb/test_samples.csv", index_col="id")
model = joblib.load("Labb/model.pkl")

df_proba = pd.DataFrame(
    model.predict_proba(df_test_samples.drop("cardio", axis=1)),
    columns=("probability class 0", "probability class 1"),
)
df_predict = pd.DataFrame(
    model.predict(df_test_samples.drop("cardio", axis=1)), columns=["prediction"]
)
df_prediction = df_proba.merge(
    df_predict,
    left_index=True,
    right_index=True,
)

df_prediction.to_csv("Labb/prediction.csv", index=True)
