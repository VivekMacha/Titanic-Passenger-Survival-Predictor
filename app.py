import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chances in Titanic")

pclass = st.slider("Enter Passenger class",1,3)
sibsp = st.slider("Enter Passenger total number of Sibiling and Spouse",1,8)
parch = st.slider("Enter Passenger total number of Paremt and Child",0,6)
fare = st.number_input("Enter the Passenger Fare")
embarked = st.selectbox("Enter Passenger embarked",['Cherbourg','Southampton','Queenstown'])
sex = st.selectbox('Enter Passenger Gender',['male','female'])

df = pd.DataFrame({
    'Pclass':[pclass],
    'Sex':[sex],
    'SibSp':[sibsp],
    'Parch':[parch],
    'Fare':[fare],
    'Embarked':[embarked]
})

st.write(df)

model = load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    labelencoder = pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehotencoder = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


df['Sex'] = labelencoder.transform(df['Sex'])
embarked = onehotencoder.transform(df[['Embarked']])

embarked = pd.DataFrame(embarked,columns=onehotencoder.get_feature_names_out())

df = pd.concat([df.drop(columns=['Embarked']),embarked],axis=1)
df[['Pclass','SibSp','Parch','Fare']] = scaler.transform(df[['Pclass','SibSp','Parch','Fare']])

y = model.predict(df)

predicted = y[0][0]

def chance(predicted):
    if predicted > 0.5:
        return "The Passenger will be survived"
    else:
        return "The Passenger wont surive the journey"

if st.button("predict Survival Chance"):
    st.write('Probability of Passenger Survival Chances:',predicted)
    st.write(chance(predicted))
