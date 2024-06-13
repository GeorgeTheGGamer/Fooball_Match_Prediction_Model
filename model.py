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

rf.fit(train[predictors], train["target"])

preds = rf.predict(test[predictors])

from sklearn.metrics import accuracy_score
#Percentsge of the time tht the prediction was accurate 

acc = accuracy_score(test["target"],preds)

print(acc)
# 0.61 answer, is accuarte 61% of the time 

combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
#Create a cross tab using pandas 

print(pd.crosstab(index=combined["actual"], columns=combined["prediction"]))
#141 times correct, 76 times not correct  
#Revise accuracy metric for wins 

from sklearn.metrics import precision_score

precision_score(test["target"], preds)
#Presision was only 47% 
#Improve this factor for precision

#Next step, improving precision and rolling averages 

#create more predictors

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City")
#Will give the single group from the data 
#we want to compute rolling avergages 
#basically taking form into account 

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    #closed left gets rid of using future predictions in the prediction, you only want the past 3 weeks for form
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    #removes rows with missing values, most prediction models cant use missing values 
    return group
    #finally return group and close the function


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
#adds the wod rollin gto each colum name 

new_cols

rolling_averages(group, cols, new_cols)
#Can pass into the algorithm to improve algorithmn

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

matches_rolling = matches_rolling.droplevel('team')
#drops the unessesary level from the dta frame 
#each row has an index where you have to call a row 
#Alot of values in the index is repeating 

matches_rolling.index = range(matches_rolling.shape[0])
#and now there are unique values for each index, for this ot work you need to have specific values for each row of the date frame.

#Retraining the data model
#using the new predictors now 

def make_predicitons(data, predictors):
#time to take previous code and finally train the full model 
    
    #split training and test data 
    train = matches[matches["date"] < "2022-01-01"]
    test = matches[matches["date"] > "2022-01-01"]
    #the training set must be before the first of january 2022

    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    #Create the predictions
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    #The combine the predictions and the actuals together
    precision = precision_score(test["target"], preds)
    #Then caluclate the precision 
    return combined, precision
    #Then return the combined and the precison 

    #All put in s function so it is easier to use 

combined, precision = make_predicitons(matches_rolling, predictors + new_cols)

precision
#Precison is now 62.5%, which is a considerable increase from the previous 40%

combined =combined.merge(matches_rolling[["date", "tean", "opponent", "result"]], left_index=True, right_index=True)
#Pandas will look in the combined data frame, it will find the corresponeding index in matches rolling and then will merge the row based on that

print(combined)
#there is now additional information needed to know what the oppenent and the team playing to know if the information is true.

#Combining home and away matches
#we have data for home and away matches, can comine the data together


class MissingDict(dict):
    _missing_ = lambda self, key: key

map_values = {"Brighton and Hove Albion" : "Brighton",
              "Manchester United" : "Manchester Utd",
              "Newcastle United" : "Newcastle Utd",
              "Tottenham Hotspur" : "Tottenham",
              "West Ham United" : "West Ham",
              "Wolverhampton Wanderers": "Wolves"
              }

#Missing dictionary replaces long names with the shorter ones 
#If you pass brentford in it will pass in brentford

mapping = MissingDict(**map_values)

mapping["Arsenal"]
#Prints Arsenal

mapping["Tottenham Hotspur"]
#Prints Tottenham

combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined, left_on=["date","new_team"], right_on = ["date", "opponent"])

merged
#This basically makes a new dataframe where you can see where the predictions line up and where it doesn't.

#Add more data for more accuracy






















