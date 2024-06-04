import pandas as pd 
#Use the pandas library to read the csv matches

matches = pd.read_csv("matches.csv", index_col=0)
#Index column is the column with 1,2,3,4,5,6,7,8,9... , indx column in pandas

matches.head()
#Prints the first few rows of the dataframe 

matches.shape
#prints the number  of rows and columns 

# 38 matches played and 20 teams spanned accross two seasons
# 38 * 20 * 2 = 1520
# This doees not line up with the 1389 columns

matches["team"].value_counts()
#Help figure out how many matches for each team in the data
#Due to relegation, different teams come up

matches[matches["team"] == "liverpool"]
#There is a missing season for liverpool

#Cleaning the data 

matches.dtypes
#The type of data in each column 
#Machine learning models can only work with float64 or int64
#Not objects

#Use datetime for data since the machine learning model can interpret it 

matches ["date"] = pd.to_datetime(matches["date"])

matches

#Time to make the predictors as the base for the model

#Chnging the home and away games to integers so that the model can interpret it
matches["venue_code"] = matches["venue"].astype("category").cat.codes
#coverts to a categorical data type in pandas because theres only two unique valus in the column and then converts into integers using .cat.codes
#converts from string to categories and then categories to integers


matches["opp_code"] = matches["opponent"].astype("category").cat.codes
#same as previous but with the opponent column

matches["hour"] = matches["time"].str.replace(":.+","",regex=True).astype("int")
#time is hour : minute, you want to replace the : and the minutes
#Need to use a number as input for a machine learning model

matches ["day_code"] = matches["date"].dt.dayofweek


matches["target"] = (matches["result"] == "W").astype("int")
#You want to check if the team has won or not#
#For more complex you can code as 1, 0 ,2 for win draw and loss for more complexity but this is the more simple way around 

#Time to train the machine learing model 

from sklearn.ensemble import RandomForestClassifier
#type of machine learning mdoel that can pick up non lineararitys in the data 
#Linear model cannot pick up the differences while a random forset can 

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10,random_state=1)
#Pass in parameters, n_estimators is the number of indviual decision trees you want ot train
#a random forest is series of decision trees  but each decison trees have differnt parameters
#the higher the number the longer it will take to run but the more accurate it would be 
#min smaples split is the number of samples you want to have in the leaf of a decision tree before you split the leaf 
# the higher the less likely to over fit but lower accuracy on training data
#set random state, means that if it is ran multiple times you would get the dame resuly since there is alot of randomness

#split training and test data 

train = matches[matches["date"] < "2022-01-01"]
test = matches[matches["date"] > "2022-01-01"]
#the training set must be before the first of january 2022

predictors = ["venue_code","opp_code","hour","day_code"]
#These are the predictors that will be used to train the machine learning model 























