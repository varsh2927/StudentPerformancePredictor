import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    df['passed'] = df['passed'].map({'yes': 1, 'no': 0})
    df = df.dropna()
    return df

def train_model(df):
    X = df.drop('passed', axis=1)
    y = df['passed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model

def get_user_input():
    print("\nPlease enter student details for prediction:")

    while True:
        gender = input("Gender (male/female): ").strip().lower()
        if gender in ['male', 'female']:
            gender_num = 0 if gender == 'male' else 1
            break
        else:
            print("Invalid input. Please enter 'male' or 'female'.")

    while True:
        try:
            study_hours = float(input("Study hours per day (e.g., 8): "))
            if study_hours < 0:
                print("Study hours cannot be negative.")
            else:
                break
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            attendance = float(input("Attendance percentage (0-100): "))
            if 0 <= attendance <= 100:
                break
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    return [gender_num, study_hours, attendance]

def predict_pass_fail(model, features):
    pred = model.predict([features])
    return "Pass" if pred[0] == 1 else "Fail"

if __name__ == "__main__":
    data_path = "../data/student_data.csv"
    df = load_data(data_path)
    df = preprocess_data(df)
    model = train_model(df)

    user_features = get_user_input()
    result = predict_pass_fail(model, user_features)
    print(f"\nPrediction result: {result}")
