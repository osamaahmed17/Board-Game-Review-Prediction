import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)
print(pandas.__version__)
print(matplotlib.__version__)
print(seaborn.__version__)
print(sklearn.__version__)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#For Loading the data
games= pandas.read_csv("games.csv")

#Print the names of the columns in the game
print(games.columns)
print(games.shape)

#Making a histogram of all the ratings in the average_rating column
plt.hist(games["average_rating"])
plt.show()

#For the printing the first row of all the zero rating games
print(games[games["average_rating"]==0].iloc[0]) #The iloc is indexing by location

#Print the first row of the games with score greater then the zero
print(games[games["average_rating"]>0].iloc[0])

#Remove any row without user reviews
games = games[games["users_rated"]>0]

#Remove any row with missing value in it
games= games.dropna(axis=0)

#Making a histogram of all the Average rating
plt.hist(games["average_rating"])
plt.show()

print(games.columns)

#Making a corelation Matrix
corrmat= games.corr()
fig = plt.figure(figsize=(12,9)) #Part of panda dataframe and figure is provided to look nice
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()

#Get all the columns from the dataframe
columns=games.columns.tolist()

#Filter the column to remove that which are not required
columns= [c for c in columns if not c in ["bayes_average_rating","average_rating","type","name","id"]]

#Store the variable we will be doing prediction upon
target="average_rating"

#Generate training and testing datasets
from sklearn.model_selection import train_test_split
#Genarating the training dataset
train= games.sample(frac=0.8, random_state=1)
#We gonna select not anyhting in the training set and we will put on test set
test = games.loc[~games.index.isin(train.index)]

#Print shapes of each sub sets
print(train.shape)
print(test.shape)

#Import the linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Initialize the linear model class
LR=LinearRegression()

#Fit the model to training data
LR.fit(train[columns],train[target])

#For Generating the predictions for testing set
predictions=LR.predict(test[columns])

#Compute Error between our predicted values and actual values
mean_squared_error(predictions,test[target])

#Import the Random Forest Model
from sklearn.ensemble import RandomForestRegressor

#Initializing the Model
RFR= RandomForestRegressor(n_estimators=100, min_samples_leaf=10,random_state=1)

#Fit to the data
RFR.fit(train[columns],train[target])

#For Making the Prediction
predictions=RFR.predict(test[columns])

#Compute Error between our predicted values and actual values
mean_squared_error(predictions,test[target])

test[columns].iloc[0]

#Making predictions through both models
rating_LR=LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR=RFR.predict(test[columns].iloc[0].values.reshape(1,-1))

print(rating_LR)
print(rating_RFR)

test[target].iloc[0]