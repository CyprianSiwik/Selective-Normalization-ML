# UCI Adult dataset loader
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AdultDataset(Dataset):
    def __init__(self, dataframe, label_encoder, scaler, is_train=True):
        self.data = dataframe
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.is_train = is_train

        self.X = self.data.drop('income', axis=1)
        self.y = self.label_encoder.transform(self.data['income'])

        # Scale numerical features only
        numeric_columns = self.X.select_dtypes(include=['int64', 'float64']).columns
        self.X[numeric_columns] = self.scaler.transform(self.X[numeric_columns])

        # Convert all to tensors
        self.X = torch.tensor(self.X.values, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_adult_dataloaders(batch_size=64, test_size=0.2, shuffle=True):
    """
    Load and preprocess the UCI Adult dataset.

    Returns:
        train_loader, test_loader, label_encoder, scaler
    """
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        names=column_names,
        na_values="?",
        skipinitialspace=True
    ).dropna()

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns.drop("income")
    df = pd.get_dummies(df, columns=categorical_cols)

    # Encode the label (income)
    label_encoder = LabelEncoder()
    df['income'] = label_encoder.fit_transform(df['income'])  # '>50K' -> 1, '<=50K' -> 0

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['income'])

    # Normalize numeric features
    scaler = StandardScaler()
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.drop('income')
    scaler.fit(train_df[numeric_cols])

    # Wrap in Dataset
    train_dataset = AdultDataset(train_df, label_encoder, scaler, is_train=True)
    test_dataset = AdultDataset(test_df, label_encoder, scaler, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_encoder, scaler


if __name__ == '__main__':
    train_loader, test_loader, le, scaler = get_adult_dataloaders()

    for X_batch, y_batch in train_loader:
        print("Features:", X_batch.shape)
        print("Labels:", y_batch.shape)
        break
