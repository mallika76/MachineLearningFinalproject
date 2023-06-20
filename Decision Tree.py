#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install graphviz


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd
from sklearn.tree import export_graphviz
# Assuming you have a dataset with categorical features X_cat and labels y
Data=pd.read_csv('bird1.csv')

df=Data.drop(labels=['RecordID','RemainsofwildlifesenttoSmithsonian','Altitudebin','AircraftType','Remainsofwildlifecollected','Name','AircraftModel','WildlifeNumberstruck','WildlifeNumberStruckActual','FlightDate','AircraftNumberofengines','Remarks','CostTotal $','Feetaboveground','IsAircraftLarge'],axis=1)

# X_cat should be a 2D array where each row represents a data instance and each column represents a categorical feature
X=df.drop(columns=['Airlinecrash'])

# y should be a 1D array containing the corresponding labels
y=df['Airlinecrash']

# Encode categorical features
X_encoded = X.copy()
for column in X.columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
tree = DecisionTreeClassifier()

# Train the classifier
tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree.predict(X_test)


# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import graphviz

dot_data = export_graphviz(tree, out_file=None, feature_names=X_encoded.columns, class_names=['NO', 'Yes'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Save the visualization as a file (optional)
graph.view()  # Display the decision tree in a viewer (e.g., Graphviz)


# In[14]:





# In[ ]:




