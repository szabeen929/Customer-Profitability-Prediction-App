
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

traveldata=pd.read_csv("C:/Users/sarwa/OneDrive/Documents/SPU/DS599/winter/travel.csv")

df=traveldata.interpolate()

df['profitability'] = 1 

df.loc[df['NumberOfTrips']<5,'profitability'] = 0
#convert categoricals to numeric
d = {'Self Enquiry':1}
df['TypeofContact']=[d.get(x,2) for x in df.TypeofContact]
d1={'Salaried':1,'Small Business':2, 'Large Business':3}
df['Occupation']=[d1.get(x,4) for x in df.Occupation]
d3={'male':1}
df['Gender']=[d3.get(x,2) for x in df.Gender]
d4={'Executive':1,'Manager':2,'Senior Manager':3,'AVP':4}
df['Designation']=[d4.get(x,5) for x in df.Designation]
d5={'Married':1,'Divorced':2,'Single':3}
df['MaritalStatus']=[d5.get(x,4) for x in df.MaritalStatus]
d6={'Basic':1,'Deluxe':2,'Standard':3,'Super Deluxe':4}
df['ProductPitched']=[d6.get(x,5) for x in df.ProductPitched]

df=df[['MonthlyIncome','TypeofContact','Occupation','CityTier','PreferredPropertyStar','profitability']]

X = df.drop('profitability', axis=1)
Y = df['profitability']

x_train, x_test, y_train, y_test=train_test_split(X, Y, random_state=1)

    
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

pickle.dump(rf_model, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2000, 2, 3, 1, 4]]))