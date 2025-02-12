import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_excel(r'rain data.xlsx')
print(df.head())

# Separate independent and dependent features
inputs = df.drop('rain', axis="columns")
output = df['rain']

# Label encode categorical column
le_places = LabelEncoder()
inputs['places_le'] = le_places.fit_transform(inputs['places'])

# Drop the original 'places' column after encoding
inputs = inputs.drop('places', axis="columns")
print(inputs.head())

# Save processed data
inputs.to_excel(r'rainy.xlsx', index=False)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(inputs, output)

scr = model.score(inputs,output)
print(scr)

#rediPct
op = model.predict([[35,51,22,965,4,0]])
print("WILL IT RAIN?" , op)


