import argparse

import numpy as np
import pandas as pd

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--data-dir", type=str, default="./data")
    args = args.parse_args()

    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"
    ]

    df_train = pd.read_csv(f"{args.data_dir}/adult/adult.data", names=cols)
    df_test = pd.read_csv(f"{args.data_dir}/adult/adult.test", names=cols, skiprows=1)

    # Filtering ? value
    for j in range(df_train.shape[1]):
        df_train = df_train[df_train.iloc[:, j] != ' ?']
        df_test = df_test[df_test.iloc[:, j] != ' ?']

    num_train = df_train.shape[0]
    num_test = df_test.shape[0]
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    print(f'train_size: {num_train}, test_size: {num_test}')

    df['age'] = pd.cut(df['age'], bins=5)
    df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=5)
    df['fnlwgt'] = pd.cut(np.log(df['fnlwgt']), bins=5)
    
    for col in ['capital-gain', 'capital-loss']:
        df[col] = (df[col] == 0.0).values.astype(int)
    df.drop(columns=['education-num'], inplace=True) # duplicate with "education"

    df['sex'] = (df['sex'] == 'Male').values.astype(int)
    
    # df['salary'] = {"<=50K", "<=50K.", ">50K", ">50K."}
    df['label'] = (
        df['salary'].apply(lambda x: x == '>50K' or x == '>50K.')
    ).values.astype(int)
    del df['salary']
    
    df_dummy = pd.get_dummies(df, dtype=int)

    df_train = df_dummy.iloc[:num_train, :]
    df_test = df_dummy.iloc[num_train:, :]

    print(f"train_size: {df_train.shape}, test_size: {df_test.shape}")

    df_train.to_pickle(f"{args.data_dir}/adult/adult_train.pkl")
    df_test.to_pickle(f"{args.data_dir}/adult/adult_test.pkl")

