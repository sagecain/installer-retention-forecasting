# pip install pandas numpy scikit-learn matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco-Customer-Churn.csv')
assert os.path.exists(DATA_PATH), f"Missing dataset at {DATA_PATH}. Download from Kaggle and place it there."

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

# Basic cleanup
df = df.copy()
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = (df['Churn'].astype(str).str.strip().str.lower() == 'yes').astype(int)
df = df.dropna()

X = df.drop(columns=['Churn'])
y = df['Churn']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def evaluate(model, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else preds
    print(f"\n{name}")
    print(" accuracy :", accuracy_score(y_test, preds):.4f)
    print(" precision:", precision_score(y_test, preds):.4f)
    print(" recall   :", recall_score(y_test, preds):.4f)
    print(" f1       :", f1_score(y_test, preds):.4f)
    try:
        print(" roc_auc  :", roc_auc_score(y_test, proba):.4f)
    except Exception:
        pass

logreg = Pipeline([('prep', preprocess), ('clf', LogisticRegression(max_iter=200))])
rf     = Pipeline([('prep', preprocess), ('clf', RandomForestClassifier(n_estimators=400, random_state=42))])

logreg.fit(X_train, y_train); evaluate(logreg, "Logistic Regression")
rf.fit(X_train, y_train);     evaluate(rf, "Random Forest")

# ROC curve for RF
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title('ROC Curve - Random Forest')
plt.tight_layout()
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results'), exist_ok=True)
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'roc_curve.png'))
print("\nSaved ROC curve to results/roc_curve.png")
