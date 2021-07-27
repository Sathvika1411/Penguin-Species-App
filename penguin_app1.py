# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.
# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)
# Display the first five rows of the DataFrame
df.head()
# Drop the NAN values
df = df.dropna()
# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})
# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})
# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})
# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']
# Slit the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)
# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)
# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)
#Create a function prediction()
@st.cache()
def prediction(model,island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g,sex):
	species=model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g,sex]])
	species=species[0]
	if species==0:
		return "Adelie"
	elif species==1:
		return "Chinstrap"
	elif species==2:
		return "Gentoo"

#Design the ML Web App
st.title("PENGUIN SPECIES PREDICTION ML WEB APP")

b_len = st.sidebar.slider("BILL LENGTH", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
b_depth = st.sidebar.slider("BILL DEPTH", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
f_len = st.sidebar.slider("FLIPPER LENGTH", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
bm_g = st.sidebar.slider("BODY MASS IN GRAMS", float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))

# Add a select box in the sidebar with the 'Classifier' label.
# Also pass 3 options as a tuple ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier').
# Store the current value of this slider in the 'classifier' variable.
#selectboxes
sex=st.sidebar.selectbox('gender',('male','female'))
if sex=='male':
	s=0
else:
	s=1

island=st.sidebar.selectbox('island',('Biscoe','Dream','Torgersen'))
if island=='Biscoe':
	isl=0
elif island=='Dream':
	isl=1
else:
	isl=2

classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
# When the 'Predict' button is clicked, check which classifier is chosen and call the 'prediction()' function.
# Store the predicted value in the 'species_type' variable accuracy score of the model in the 'score' variable. 
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
if st.sidebar.button("PREDICT"):
	if classifier=='Support Vector Machine':
		species_type=prediction(svc_model,isl,b_len,b_depth,f_len,bm_g,s)
		score=svc_model.score(X_train,y_train)

	elif classifier=='LogisticRegression':
		species_type=prediction(log_reg,isl,b_len,b_depth,f_len,bm_g,s)
		score=log_reg.score(X_train,y_train)

	else:
		species_type=prediction(rf_clf,isl,b_len,b_depth,f_len,bm_g,s)
		score=rf_clf.score(X_train,y_train)

	st.write("Species Prdicted : ",species_type)
	st.write("Accuracy score of this",classifier,"model is : ",score)
