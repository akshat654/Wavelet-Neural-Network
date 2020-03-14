# Importing the libraries
import pandas as pd 
import numpy as numpy
import matplotlib.pyplot as plt 


# Loading the csv file
print('Loading the data...')
data = pd.read_csv('train.csv')


# Renaming the column
data = data.rename(columns={'electricity_consumption':'demand'})


# Dropping the ID Column
data.drop(['ID'], axis=1, inplace=True)


# Some attributes of data
n_examp, n_feat = data.shape


# Change the datetime column(a str) to datetime type
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')


# Get the range of time
min_time = min(data['datetime'])
max_time = max(data['datetime'])


# Make new columns of day, month, year, hour
data['year'] = data.datetime.dt.year
data['month'] = data.datetime.dt.month
data['day'] = data.datetime.dt.day
data['hour'] =  data.datetime.dt.hour


# Rearranging the columns
data = data[['datetime','year', 'month', 'day', 'hour',
			'temperature', 'var1', 'pressure', 'windspeed', 'var2','demand']]


'''

Important Observation from the training data

The data contains continuous data for day 1-day 23 of each month
rest of the days have not been given

'''

# Dropping the datetime column since its redundant
data.drop(['datetime'], axis=1, inplace=True)



# Encoding var2 column with A:1 and B/C:0
data.var2.replace({'A':1, 'B': 0, 'C': 0}, inplace=True)


#Normalizing the columns
max_dem = max(data['demand'])
min_dem = min(data['demand'])
for col in data.columns:
    data[col] = (data[col] - min(data[col]))/(max(data[col]) - min(data[col]))

print(max_dem, min_dem)


data.to_csv('data.csv')
