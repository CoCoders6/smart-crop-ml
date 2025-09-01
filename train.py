import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# -------------------------------
# Create synthetic dataset
# -------------------------------
rows = []

for ph in [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]:
    for n in [50, 70, 90, 110, 130, 150, 170, 190, 200]:
        for rain in range(0, 11):  # realistic rainfall 0-10 mm/h
            # Crop assignment rules
            if ph > 6.5 and rain > 5 and n > 100:
                crop = 'Paddy'
            elif n > 150 and rain < 5:
                crop = 'Maize'
            elif ph < 6.0 and rain < 3:
                crop = 'Millets'
            elif ph >= 6.0 and ph <= 7.0 and rain < 4 and n <= 120:
                crop = 'Wheat'
            elif ph > 7.0 and rain > 2:
                crop = 'Barley'
            elif rain > 7:
                crop = 'Sugarcane'
            elif n > 130 and ph < 7.0:
                crop = 'Soybean'
            else:
                crop = 'Groundnut'
            rows.append({'pH': ph, 'N': n, 'rain': rain, 'crop': crop})

# Create DataFrame
df = pd.DataFrame(rows)
X = df[['pH', 'N', 'rain']]
y = df['crop']

# -------------------------------
# Train model
# -------------------------------
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

# -------------------------------
# Save model
# -------------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/crop_recommender.pkl')
print("Model saved to models/crop_recommender.pkl")
