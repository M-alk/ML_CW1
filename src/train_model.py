import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.base import clone
from catboost import CatBoostRegressor

RANDOM_STATE = 42
TARGET = "outcome"


def main():
    train_path = os.path.join("data", "CW1_train.csv")
    test_path = os.path.join("data", "CW1_test.csv")
    out_dir = "outputs"
    out_path = os.path.join(out_dir, "CW1_submission_k22056537.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    os.makedirs(out_dir, exist_ok=True)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop low-signal columns
    drop_cols = ["a5", "a6", "a7", "a9", "b5", "b6", "b7", "b9"]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    y = train_df[TARGET]
    X = train_df.drop(columns=[TARGET])

    # Features used in final run
    for frame in [X, test_df]:
        frame["depth_sq"] = frame["depth"] ** 2
        frame["depth_x_table"] = frame["depth"] * frame["table"]
        frame["log_price"] = np.log1p(frame["price"])

        # Interaction terms that helped 
        frame["b1_x_a1"] = frame["b1"] * frame["a1"]
        frame["b3_b1_a1"] = frame["b3"] * frame["b1"] * frame["a1"]

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Target encode categoricals, pass through numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", TargetEncoder(random_state=RANDOM_STATE), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Manual CV loop keeps CatBoost + sklearn stable
    # depth=3, l2_leaf_reg=30, subsample=0.5, lr=0.01, iters=3000

    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    for train_idx, test_idx in kf.split(X, y):
        pp = clone(preprocessor)
        X_tr = pp.fit_transform(X.iloc[train_idx], y.iloc[train_idx])
        X_te = pp.transform(X.iloc[test_idx])
        cb = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.01,
            depth=3,
            l2_leaf_reg=30,
            subsample=0.5,
            random_seed=RANDOM_STATE,
            verbose=0,
        )
        cb.fit(X_tr, y.iloc[train_idx])
        pred = cb.predict(X_te)
        fold_scores.append(r2_score(y.iloc[test_idx], pred))

    print(f"CV R² = {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    X_all = preprocessor.fit_transform(X, y)
    X_test = preprocessor.transform(test_df)

    final_model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.01,
        depth=3,
        l2_leaf_reg=30,
        subsample=0.5,
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    final_model.fit(X_all, y)
    preds = final_model.predict(X_test)

    pd.DataFrame({"yhat": preds}).to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Predictions: {len(preds)}")
    print("Preview:")
    print(pd.read_csv(out_path).head())


if __name__ == "__main__":
    main()