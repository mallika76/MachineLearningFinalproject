#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# In[3]:


Airlinecrash = pd.read_csv('bird1.csv')
print(Airlinecrash.head())


# In[4]:


columns_to_drop = ['RecordID', 'RemainsofwildlifesenttoSmithsonian', 'Altitudebin', 'AircraftType', 'Remainsofwildlifecollected',
                   'Name', 'AircraftModel', 'WildlifeNumberstruck', 'WildlifeNumberStruckActual', 'FlightDate', 'AircraftNumberofengines',
                   'Remarks', 'CostTotal $', 'Feetaboveground', 'IsAircraftLarge']
Airline = Airlinecrash.drop(columns_to_drop, axis=1)
Airline['Airlinecrash'].replace(['No', 'No', 'Yes'], inplace=True)
print(Airline['Airlinecrash'].value_counts())
Airline.dropna(inplace=True)
Airline.boxplot()


# In[5]:


Airline_encoded = pd.get_dummies(Airline, drop_first=True)


# In[6]:


X = Airline_encoded.drop('Airlinecrash_Yes', axis=1)
y = Airline_encoded['Airlinecrash_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33333)


# In[7]:


model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
pred_knn = model_knn.predict(X_test)
result_knn = confusion_matrix(y_test, pred_knn)
accuracy = accuracy_score(y_test, pred_knn)
print("Accuracy:", accuracy)
print(result_knn)
print(classification_report(y_test, pred_knn))


# In[8]:


print(result_knn)


# In[9]:


labels = ['No', 'Yes']
fig, ax = plt.subplots()
im = ax.imshow(result_knn, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels, xlabel='Predicted', ylabel='True',
       title='Confusion Matrix') 
plt.show()


# In[10]:


feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': [0] * len(X.columns)})
print(feature_importance)


# In[11]:


crash_counts = Airlinecrash['Conditionsky'].value_counts()

# Get the unique sky condition categories
sky_conditions = crash_counts.index

custom_colors = ['blue', 'green', 'red']

# Plotting
plt.bar(sky_conditions, crash_counts, color=custom_colors)
plt.xlabel('Sky Condition')
plt.ylabel('Airline Crash')
plt.title('SKYCONDITION VS AIRLINECRASH')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.show()

import matplotlib.pyplot as plt

# Count the number of airline crashes for each sky condition category
crash_counts = Airlinecrash['WhenPhaseofflight'].value_counts()

# Get the unique sky condition categories and their corresponding counts
WhenPhaseofflight = crash_counts.index
counts = crash_counts.values

# Define custom colors for the pie slices
custom_colors = ['blue', 'green', 'red', 'yellow', 'pink','cyan']

# Plotting
plt.pie(counts, labels=WhenPhaseofflight, colors=custom_colors, autopct='%1.1f%%')
plt.title('WhenPhaseofflight')
plt.axis('equal')  # Ensure pie is drawn as a circle
plt.show()



# In[12]:


import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'hello_cleaned' containing the relevant data

# Get the unique sky condition categories
WildlifeSize = Airlinecrash['WildlifeSize'].unique()

# Count the number of airline crashes for each sky condition category
crash_counts = Airlinecrash['WildlifeSize'].value_counts()

# Create a list of x-coordinates for each sky condition
x_coords = [i for i in range(len(WildlifeSize))]

# Get the corresponding crash counts for each sky condition
y_coords = [crash_counts[condition] for condition in WildlifeSize]

# Plotting
plt.scatter(x_coords, y_coords)
plt.xticks(x_coords, sky_conditions, rotation=45)
plt.xlabel('Wild Life Size')
plt.ylabel('Airline Crash')
plt.title('Wild Life size VS AIRLINECRASH')
plt.show()


# In[13]:


Airline.head()


# In[ ]:




