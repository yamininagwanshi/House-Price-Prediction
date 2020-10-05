import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from guietta import _, Gui, Quit
from sklearn.linear_model import LinearRegression

 
df = pd.read_csv('homeprices.csv')
#print(df.head())

#Independent variable
X = (df.iloc[:,0].values).reshape(-1,1)
#Dependent variable
y = df.iloc[:,1].values

#print(X,y) 
#plt.scatter(X,y)
#plt.show()

#Split Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

lr = LinearRegression()
lr.fit(X_train,y_train)   #fit the model
y_pred = lr.predict(X_test)   #predict the new data

coefficient = lr.coef_
#print(coefficient)

inter_cept = lr.intercept_
#print(inter_cept)

#User interface
gui = Gui(
    ["Area in Sq ft: ",_, '__area__'],
    [_,['Predict '], _],
    ["House Price: ", _, 'price'],
    [_,Quit, _]
    )
def price_pred(gui):
    x = float(gui.area)
    y = 159.3220339 * x + 95338.98305084743
    gui.price = y
    
gui.Predict = price_pred

gui.run()


