#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Step 1: Import the dataset
data = pd.read_csv('bird1.csv')
hello=data.drop(labels=['RecordID','RemainsofwildlifesenttoSmithsonian','Altitudebin','AircraftType','Remainsofwildlifecollected','Name','AircraftModel','WildlifeNumberstruck','WildlifeNumberStruckActual','FlightDate','AircraftNumberofengines','Remarks','CostTotal $','Feetaboveground','IsAircraftLarge'],axis=1)

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

# Fit the Naive Bayes classifier
NB = GaussianNB()
NB.fit(mtraining_encoded, y_train)

# Make predictions
predNB1 = NB.predict(mtesting_encoded)
print(predNB1)
print(accuracy_score(mtesting['Airlinecrash'], predNB1))

import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'hello_cleaned' containing the relevant data

# Count the number of airline crashes for each sky condition category
crash_counts = hello['Conditionsky'].value_counts()

# Get the unique sky condition categories
sky_conditions = crash_counts.index

# Plotting
plt.bar(sky_conditions, crash_counts)
plt.xlabel('Sky Condition')
plt.ylabel('Airline Crash')
plt.title('SKYCONDITION VS AIRLINECRASH')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.show()


# In[3]:


hello.head()


# In[ ]:




