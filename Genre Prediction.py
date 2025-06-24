import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# ------------------------------
# 📥 Data Loaders
# ------------------------------

def parse_train_line(line):
    parts = line.strip().split(" ::: ")
    if len(parts) == 4:
        return {
            "id": parts[0],
            "title": parts[1],
            "genre": parts[2].lower(),
            "description": parts[3]
        }
    return None

def parse_test_line(line):
    parts = line.strip().split(" ::: ")
    if len(parts) == 3:
        return {
            "id": parts[0],
            "title": parts[1],
            "description": parts[2]
        }
    return None

def parse_test_solution_line(line):
    parts = line.strip().split(" ::: ")
    if len(parts) == 4:
        return {
            "id": parts[0],
            "title": parts[1],
            "genre": parts[2].lower(),
            "description": parts[3]
        }
    return None

def load_data(file_path, parser_func):
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    data = [parser_func(line) for line in lines if parser_func(line)]
    return pd.DataFrame(data)

# ------------------------------
# 🧠 Model Pipeline
# ------------------------------

def build_pipeline():
    tfidf = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2
    )
    
    xgb = XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        num_class=20,  # adjust based on your dataset
        use_label_encoder=False,
        random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', xgb)
    ])
    return pipeline

# ------------------------------
# 📈 Main Workflow
# ------------------------------

if __name__ == "__main__":
    train_file = "train_data.txt"
    test_file = "test_data.txt"
    test_solution_file = "test_data_solution.txt"

    # Load data
    train_df = load_data(train_file, parse_train_line)
    test_df = load_data(test_file, parse_test_line)
    test_solution_df = load_data(test_solution_file, parse_test_solution_line)

    print("🎬 Genre Distribution (Training):")
    print(train_df['genre'].value_counts(), "\n")

    # Encode genres
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(train_df["genre"])

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        train_df["description"], y_encoded,
        test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Build pipeline
    pipeline = build_pipeline()

    # Hyperparameter tuning (optional but recommended!)
    param_grid = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [6, 8, 10],
        'clf__learning_rate': [0.05, 0.1]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"🔍 Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # Validation
    val_preds = best_model.predict(X_val)
    print("📊 Validation Performance:")
    print(classification_report(y_val, val_preds, target_names=label_encoder.classes_))
    print(f"✅ Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}\n")

    # Final model on full training data
    best_model.fit(train_df["description"], y_encoded)

    # Predict on test set
    test_preds = best_model.predict(test_df["description"])
    test_df["predicted_genre"] = label_encoder.inverse_transform(test_preds)

    # Evaluate against test solution
    y_true = label_encoder.transform(test_solution_df["genre"])
    y_pred = best_model.predict(test_solution_df["description"])

    print("📉 Test Set Evaluation (using Test Solution):")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print(f"🎯 Test Set Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

    # Show predicted test genres
    print("🔮 Predicted Test Genres:")
    print(test_df[["id", "title", "predicted_genre"]])
