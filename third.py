# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

# Ignore Warnings
warnings.filterwarnings('ignore')

# -----------------------
# Load and Preprocess Data
# -----------------------
def load_data(url):
    """Load dataset from URL."""
    data = pd.read_csv(url)
    return data

def preprocess_data(data):
    """Preprocess the dataset: Encode categorical data and split into features and target."""
    x = data[['R&D Spend', 'Administration', 'Marketing Spend', 'State']].values
    y = data['Profit'].values

    # Label Encoding and OneHot Encoding for 'State'
    label_encoder = LabelEncoder()
    x[:, 3] = label_encoder.fit_transform(x[:, 3])

    transformer = ColumnTransformer(
        transformers=[("OneHot", OneHotEncoder(), [3])], 
        remainder='passthrough'
    )

    x = transformer.fit_transform(x).astype(float)
    return x, y

# -----------------------
# Model Development
# -----------------------
def train_model(x, y, test_size=0.2, random_state=0):
    """Split data, train the model, and return it with accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(x_train, y_train)

    # Model Accuracy
    pred_values = model.predict(x_test)
    accuracy = r2_score(y_test, pred_values) * 100
    print(f'Model Accuracy: {accuracy:.2f}%')

    return model

# -----------------------
# User Input and Prediction
# -----------------------
def get_user_input():
    """Get user input for startup details."""
    california = int(input('Is startup in California? (1 = Yes, 0 = No): '))
    florida = int(input('Is startup in Florida? (1 = Yes, 0 = No): '))
    newyork = int(input('Is startup in New York? (1 = Yes, 0 = No): '))
    
    r_d = float(input('Enter Research & Development Spend: '))
    admin = float(input('Enter Administration Spend: '))
    market = float(input('Enter Marketing Spend: '))

    user_data = pd.DataFrame({
        'california': [california],
        'florida': [florida],
        'newyork': [newyork],
        'r_d': [r_d],
        'admin': [admin],
        'market': [market]
    })

    return user_data

def make_prediction(model, user_data):
    """Make profit prediction using the trained model."""
    prediction = model.predict(user_data)
    print(f"\nEstimated Profit: ${prediction[0]:,.2f}")

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    # Load and preprocess the data
    url = 'https://raw.githubusercontent.com/yash240990/Python/master/Startups_Data.csv'
    data = load_data(url)
    x, y = preprocess_data(data)

    # Train the model
    model = train_model(x, y)

    # Get user input and make prediction
    user_data = get_user_input()
    make_prediction(model, user_data)
