import joblib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
df = sns.load_dataset("penguins")
df.head()
df.shape
df.isna().sum()
df = df.dropna()
df.shape
df['sex'] = df['sex'].replace({'Male' : 1, 'Female': 0})
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['species']
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print(classification_report(y_test, y_pred))
joblib.dump(forest, "model.joblib")
print("model.joblib")