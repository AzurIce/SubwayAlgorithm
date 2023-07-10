import pandas as pd
import os

df = pd.read_csv('new_traffic_data.csv')
# data.append([neighborhoodSize,TEntries,TExits,Year,Month,Day,Hour,int(holiday),neighborhoodSize])
entries = df['Entries']
exits = df['Exits']
id = df['Unique ID']
year = df['Year']
month = df['Month']
day = df['Day']
hour = df['Hour']
holiday = df['isHoliday']
struct = df['Str']
# Unique ID,Structure,Neighborhood,Entries,Exits,Year,Month,Day,Hour,isHoliday,Neighborhood Size