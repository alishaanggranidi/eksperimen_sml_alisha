from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(data, target_column, save_path, file_path):
    numeric_features = data.select_dtypes(include=['float64','int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns.drop(target_column)

    # membuat df kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)

    # cave nama kolom ke csv
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num',numeric_transformer, numeric_features),
            ('cat',categorical_transformer, categorical_features)
        ]
    )

    # split fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # split data menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # fitting dan transformasi data pada training set
    X_train = preprocessor.fit_transform(X_train)
    # transformasi data pada testing set
    X_test = preprocessor.transform(X_test)
    # save pipeline
    dump(preprocessor, save_path)

    return X_train, X_test, y_train, y_test