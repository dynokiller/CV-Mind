import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

print("Loading...")
df = pd.read_csv(r"C:\Users\Admin\Downloads\archive\Resume\Resume.csv").dropna(subset=['Resume_str', 'Category'])
texts = df['Resume_str'].tolist()
y = LabelEncoder().fit_transform(df['Category'].tolist())
X = TfidfVectorizer(max_features=1000).fit_transform(texts) # just 1000 for speed test

print("Training on CUDA with sparse...")
model_sparse = xgb.XGBClassifier(n_estimators=100, device='cuda')
model_sparse.fit(X, y)
pred_sparse = model_sparse.predict(X)
print("CUDA sparse accuracy:", accuracy_score(y, pred_sparse))

print("Training on CUDA with dense...")
model_dense = xgb.XGBClassifier(n_estimators=100, device='cuda')
model_dense.fit(X.toarray(), y)
pred_dense = model_dense.predict(X.toarray())
print("CUDA dense accuracy:", accuracy_score(y, pred_dense))

print("Training on CPU with sparse...")
model_cpu = xgb.XGBClassifier(n_estimators=100, device='cpu')
model_cpu.fit(X, y)
pred_cpu = model_cpu.predict(X)
print("CPU sparse accuracy:", accuracy_score(y, pred_cpu))
