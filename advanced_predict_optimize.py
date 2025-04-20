import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1) Load data
df = pd.read_csv('data/student_habits_performance.csv')

# 2) Feature engineering
df['study_social_ratio'] = df['study_hours_per_day'] / (df['social_media_hours'] + 1e-6)
df['sleep_efficiency'] = df['sleep_hours'] * df['attendance_percentage'] / 100

# 3) Define target & features
target = 'exam_score'
features = df.drop(columns=['student_id', target])

# 4) Identify numeric & categorical columns
num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = features.select_dtypes(include=['object', 'bool']).columns.tolist()

# 5) Build preprocessing + model pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
pipeline = Pipeline([
    ('prep', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 6) Split data and train
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 7) Evaluate
y_pred_test = pipeline.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
cv_scores = -cross_val_score(pipeline, features, df[target], cv=3, scoring='neg_mean_absolute_error')
print(f"Test MAE: {test_mae:.2f}, Test R²: {test_r2:.2f}, CV MAE: {cv_scores.mean():.2f}")

# 8) Compute benchmarks (>90th percentile)
pct90 = df[target].quantile(0.90)
top90 = df[df[target] >= pct90]
benchmarks = {
    'study_hours_per_day': top90['study_hours_per_day'].mean(),
    'sleep_hours': top90['sleep_hours'].mean(),
    'study_social_ratio': top90['study_social_ratio'].mean(),
    'attendance_percentage': top90['attendance_percentage'].mean()
}

# 9) Predict on all & flag at-risk (<50)
df['predicted_score'] = pipeline.predict(features)
df['at_risk'] = df['predicted_score'] < 50

# 10) Recommendation function
def recommend(row):
    tips = []
    if row['at_risk']:
        if row['study_hours_per_day'] < benchmarks['study_hours_per_day']:
            tips.append(f"Boost daily study to ~{benchmarks['study_hours_per_day']:.1f}h")
        if row['attendance_percentage'] < 90:
            tips.append("Aim for ≥90% class attendance")
        if row['sleep_hours'] < 7:
            tips.append("Get 7–9h sleep nightly")
        tips.append("Reduce distractions: limit social media & Netflix during study blocks")
    else:
        tips.append("Try spaced‐repetition flashcards for retention")
        if row['study_social_ratio'] < benchmarks['study_social_ratio']:
            tips.append("Optimize study/leisure balance: use Pomodoro technique")
        if row['sleep_hours'] < 8:
            tips.append("Maintain consistent sleep schedule for peak cognition")
        tips.append("Practice with timed mock exams to build exam-taking stamina")
    return tips

df['recommendations'] = df.apply(recommend, axis=1)

# 11) Show sample
sample = df.sample(1, random_state=42).iloc[0]
print("\nSample Student Recommendation\n" + "-"*30)
print(f"Student ID: {sample['student_id']}")
print(f"Actual Score: {sample['exam_score']:.1f}, Predicted Score: {sample['predicted_score']:.1f}")
status = "AT RISK" if sample['at_risk'] else "ON TRACK"
print(f"Status: {status}\nRecommendations:")
for tip in sample['recommendations']:
    print(" -", tip)
