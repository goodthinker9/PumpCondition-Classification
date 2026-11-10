"""
CHAPTER 3 PRACTICAL ASSIGNMENT
WATER PUMP CONDITION CLASSIFICATION

This is a complete, runnable Python program based on the pump example.
It uses six supervised classification algorithms.

Class labels:
    0 = Normal
    1 = Warning
    2 = Fault

Install the required libraries in the terminal:
    pip install numpy pandas matplotlib seaborn scikit-learn
"""

# ============================================================
# PART 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# PART 2. CREATE DATASET (30 PUMP RECORDS)
# ============================================================

data = [
    # ID, Vibration, Temperature, Pressure, Speed, Operating_Hours, Condition
    [1,  1.2, 45, 5.0, 1480,  800, 0],
    [2,  1.5, 48, 4.9, 1490, 1100, 0],
    [3,  1.8, 52, 4.8, 1475, 1400, 0],
    [4,  2.0, 55, 4.7, 1500, 1700, 0],
    [5,  2.2, 57, 4.8, 1460, 2000, 0],
    [6,  1.4, 50, 5.1, 1510, 1000, 0],
    [7,  2.4, 59, 4.6, 1450, 2200, 0],
    [8,  1.7, 53, 4.9, 1495, 1500, 0],
    [9,  2.6, 61, 4.5, 1440, 2400, 0],
    [10, 1.3, 47, 5.0, 1505,  900, 0],

    [11, 3.2, 65, 4.2, 1420, 2800, 1],
    [12, 3.6, 68, 4.1, 1410, 3100, 1],
    [13, 4.0, 70, 4.0, 1400, 3400, 1],
    [14, 3.4, 66, 4.3, 1430, 2900, 1],
    [15, 4.3, 72, 3.9, 1390, 3700, 1],
    [16, 3.8, 69, 4.1, 1405, 3300, 1],
    [17, 4.6, 74, 3.8, 1380, 4000, 1],
    [18, 3.1, 64, 4.4, 1440, 2700, 1],
    [19, 4.1, 71, 3.9, 1395, 3600, 1],
    [20, 4.5, 73, 3.7, 1375, 3900, 1],

    [21, 6.0, 82, 3.2, 1340, 5000, 2],
    [22, 6.5, 86, 3.0, 1320, 5600, 2],
    [23, 7.1, 90, 2.8, 1290, 6400, 2],
    [24, 5.8, 80, 3.3, 1360, 4800, 2],
    [25, 7.5, 94, 2.6, 1270, 7100, 2],
    [26, 6.2, 84, 3.1, 1330, 5300, 2],
    [27, 8.0, 97, 2.4, 1250, 7800, 2],
    [28, 5.5, 78, 3.4, 1380, 4500, 2],
    [29, 7.0, 91, 2.7, 1300, 6700, 2],
    [30, 8.3, 99, 2.3, 1230, 8200, 2]
]

columns = [
    "Pump_ID",
    "Vibration_mm_s",
    "Temperature_C",
    "Pressure_bar",
    "Speed_RPM",
    "Operating_Hours",
    "Pump_Condition"
]

df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 WATER PUMP RECORDS")
print(df.to_string(index=False))

print("\nCLASS DISTRIBUTION")
print(df["Pump_Condition"].value_counts().sort_index().rename(
    index={0: "Normal", 1: "Warning", 2: "Fault"}
))


# ============================================================
# PART 3. EXPLORE AND VISUALIZE THE DATA
# ============================================================

print("\nDATA INFORMATION")
df.info()

print("\nSTATISTICAL DESCRIPTION")
print(df.describe())

sns.set_theme(style="whitegrid")

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Temperature_C",
    y="Vibration_mm_s",
    hue="Pump_Condition",
    style="Pump_Condition",
    s=100,
    palette="viridis"
)
plt.title("Water Pump Temperature vs Vibration")
plt.xlabel("Temperature (Celsius)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0=Normal, 1=Warning, 2=Fault")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(
    df.drop(columns=["Pump_ID"]).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix of Pump Measurements")
plt.tight_layout()
plt.show()


# ============================================================
# PART 4. SEPARATE FEATURES (X) AND TARGET (y)
# ============================================================

feature_cols = [
    "Vibration_mm_s",
    "Temperature_C",
    "Pressure_bar",
    "Speed_RPM",
    "Operating_Hours"
]

X = df[feature_cols]
y = df["Pump_Condition"]


# ============================================================
# PART 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTRAIN-TEST SPLIT")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# PART 6. DEFINE SIX CLASSIFICATION MODELS
# ============================================================

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "KNN": knn_model,
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}


# ============================================================
# PART 7. TRAIN, PREDICT, AND EVALUATE EACH MODEL
# ============================================================

results = []
predictions = {}

class_names = ["Normal", "Warning", "Fault"]

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(model_name.upper())
    print("=" * 65)

    # Train the model
    model.fit(X_train, y_train)

    # Predict test records
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    # Calculate evaluation scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    ))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ============================================================
# PART 8. COMPARE THE SIX CLASSIFICATION MODELS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")

print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1.05)
)
plt.title("Comparison of Six Classification Algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ============================================================
# PART 9. FEATURE IMPORTANCE AND DECISION TREE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} FEATURE IMPORTANCE")
    print(importance.round(3))

    plt.figure(figsize=(8, 5))
    importance.sort_values().plot(kind="barh")
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Pump Measurement")
    plt.tight_layout()
    plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(18, 9))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree for Water Pump Condition")
plt.tight_layout()
plt.show()


# ============================================================
# PART 10. PREDICT A NEW PUMP OBSERVATION
# ============================================================

new_pump = pd.DataFrame([{
    "Vibration_mm_s": 6.3,
    "Temperature_C": 85,
    "Pressure_bar": 3.0,
    "Speed_RPM": 1325,
    "Operating_Hours": 5500
}])

print("\nNEW PUMP OBSERVATION")
print(new_pump.to_string(index=False))

label_names = {
    0: "Normal",
    1: "Warning",
    2: "Fault"
}

print("\nPREDICTIONS FROM ALL SIX MODELS")

for name, model in models.items():
    predicted_class = int(model.predict(new_pump)[0])
    predicted_label = label_names[predicted_class]
    print(f"{name:15s} prediction: {predicted_label}")


# ============================================================
# END OF PRACTICAL CASE STUDY
# ============================================================

print("\nEND OF WATER PUMP CLASSIFICATION PROGRAM")
print("Note: This is an educational example, not a real pump diagnosis.")
