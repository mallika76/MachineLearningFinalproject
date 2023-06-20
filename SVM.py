#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Step 1: Import the dataset
data = pd.read_csv('bird1.csv')
hello = data.drop(labels=['RecordID', 'RemainsofwildlifesenttoSmithsonian', 'Altitudebin', 'AircraftType', 'Remainsofwildlifecollected', 'Name', 'AircraftModel', 'WildlifeNumberstruck', 'WildlifeNumberStruckActual', 'FlightDate', 'AircraftNumberofengines', 'Remarks', 'CostTotal $', 'Feetaboveground', 'IsAircraftLarge'], axis=1)

print(hello['Airlinecrash'].unique())
hello['Airlinecrash'] = hello['Airlinecrash'].replace(['value1', 'value2'], ['No', 'Yes'])
print(hello['Airlinecrash'].unique())
print(hello)
print(hello.isna().any())
print(hello.shape)

# Splitting the dataset into train and test sets
tindex = sorted(np.random.choice(hello.index, int(len(hello) * 0.3), replace=False))
mtraining = hello.loc[tindex]
mtesting = hello.loc[~hello.index.isin(tindex)]

# Combine training and testing data
combined_data = pd.concat([mtraining.drop('Airlinecrash', axis=1), mtesting.drop('Airlinecrash', axis=1)])

# Perform one-hot encoding on the combined data
combined_encoded = pd.get_dummies(combined_data)

# Split the encoded data back into training and testing sets
mtraining_encoded = combined_encoded[:mtraining.shape[0]]
mtesting_encoded = combined_encoded[mtraining.shape[0]:]

# Get the target variable
y_train = mtraining['Airlinecrash']

# Fit the SVM classifier
svm = SVC()
svm.fit(mtraining_encoded, y_train)

# Get the target variable for testing data
y_test = mtesting['Airlinecrash']

# Make predictions
pred_svm = svm.predict(mtesting_encoded)
print(pred_svm)
print(accuracy_score(y_test, pred_svm))

# Print the classification report
print(classification_report(y_test, pred_svm))



# In[ ]:




