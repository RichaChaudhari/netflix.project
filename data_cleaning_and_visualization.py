#import necessary libraries
import pandas as pd
import numpy as np

#read the file
data = pd.read_csv("netflix_titles.csv")
data.head(5)

#check columns
print(data.columns)
print("")
#print(data.info)
#print(data.describe)
#converting columns to their respective type
data = data.astype({'show_id' : object, 'type' : object, 'title' : object, 'director' : object, 'cast' : object, 
                    'country' : object, 'date_added' : object, 'rating' : object, 'duration' : object, 
                    'listed_in' : object, 'description' : object})
#check for missing data
missing_data = data.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    
#checking if the columns are in their right type
data.dtypes

#removing or replacing null values
#remove 'director' column because it has too many unknown values
data.drop(["director"], axis = 1, inplace = True)

#drop dsescription and cast column too because i won't be using that column
data.drop(['description'], axis = 1, inplace = True)
data.drop(['cast'], axis = 1, inplace = True)
data.reset_index(drop = True, inplace = True)

#dropping the missing value row in country column
data.dropna(subset = ["country"], axis = 0, inplace = True)
data.reset_index(drop = True, inplace = True)


#let's drop the date added section because we have year
data.drop(["date_added"], axis = 1, inplace = True)
data.reset_index(drop = True, inplace = True)

#fixing the 'rating' column
#to get the index of missing values
print(data[data['rating'].isnull()].index.tolist())
#get the 'title' corresponding to the rating index
print(data.loc[2191, 'title'])
#search the title to get rating online
#put the correct value
data.loc[2191, 'rating'] = "TV- 14"

#repeat the above process
print(data.loc[3470, 'title'])
data.loc[3470, 'rating'] = 'TV-MA'

print(data.loc[3407, 'title'])
data.loc[3407, 'rating'] = 'PG'

print(data.loc[4028, 'title'])
data.loc[4028, 'rating'] = 'PG-13'

#there is no rating for the remaining two movies
data = data.drop(3471)
data.reset_index(drop = True, inplace = True)
data = data.drop(3471)
data.reset_index(drop = True, inplace = True)

#data is cleaned now so i would save it.
data.to_csv("netflix.csv")

#check null values
missing_data = data.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    
#the column 'type' has only two unique values - "TV Show" or 'Movie'
dummy_var = pd.get_dummies(data['type'])
dummy_var.head()
#change column name for clarity
dummy_var.rename(columns = {'Movie' : 'type - movie', 'TV Show' : 'type TV Show'}, inplace = True)
dummy_var.head()
#merge dataframe "data" and "dummy_var"
data = pd.concat([data, dummy_var], axis = 1)
#drop original column 'type' from data
data.drop("type", axis = 1, inplace = True)
data.head(5)

#Binning data in pandas
#let's plot the histogram of years
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(data["release_year"])
#set x and y labels and title of the plot
plt.xlabel("movie released in year from 1942 to 2021")
plt.ylabel("count")
plt.title("movie year-wise bins")
plt.show()
bins = np.linspace(min(data["release_year"]), max(data["release_year"]), 4)
group_names = ['high', 'medium', 'low']
data["year bins"] = pd.cut(data["release_year"], bins, labels = group_names, include_lowest = True)
data[["release_year", "year bins"]].head(20)
#let's see the number of movies in each bins
print(data["year bins"].value_counts())

from matplotlib import pyplot
pyplot.bar(group_names, data["year bins"].value_counts())
plt.xlabel("movie release year")
plt.ylabel("counts")
plt.title("year bins")
plt.show()

#bins visualization
plt.hist(data["release_year"], bins = 3)
plt.xlabel("year")
plt.ylabel("count")
plt.title("movie released year bins")

#observation: number of movies and TV Show sharply increases after 2015

#using 3 columns - 'release_year', 'type TV Show', 'type - movie'
year = data[['release_year', 'type TV Show', 'type - movie', 'rating']]
#adding movie and TV Show release by their respective year using groupby method
release_year = year.groupby(['release_year'], as_index = False).sum()
release_year.set_index("release_year", inplace = True)
release_year.plot(kind = "area", stacked = False,alpha = 0.35, figsize = (20, 10))
plt.title("area plot movie and TV Show release in specific year")
plt.xlabel("years")
plt.ylabel("movie and TV Show release")
plt.show()
#observation: movie and TV show are increased drastically over the period of time

rating = data[['rating', 'type TV Show', 'type - movie']]
#adding movie and TV Show release by their respective year using groupby method
rating_data = rating.groupby(['rating'], as_index = False).sum()
#checking if the sum function answer correctly
rating_data.set_index("rating", inplace = True)
rating_data.plot(kind = "area", stacked = False, figsize = (20, 10))
plt.title("area plot of movie and TV Show release based on ratings")
plt.xlabel("ratings")
plt.ylabel("movie and TV Show release")
plt.show()
print(rating_data)
#observation: there is no TV Show rated 'G', 'UR', 'NC-17' or 'PG-13'.
#most of the movies and TV Shows are rated 'TV-MA'
#more TV shows are rated 'TV-Y7' and 'TV-Y' than movies

#let's make the area plot of the top 5 countries based on their release capacity
country = data[['country', 'type - movie', 'type TV Show']]
top_5_countries = country.groupby(['country'], as_index = False).sum()
#top 5 countries who released most movies
top_5_countries.sort_values(['type - movie'], ascending = False, axis = 0, inplace = True)
top5 = top_5_countries.head(5)
top5.set_index("country", inplace = True)
print(top5)
top5.plot(kind = 'area', stacked = False, figsize = (20, 10))
plt.title("area plot of top 5 countries movie-wise")
plt.xlabel("countries")
plt.ylabel("no. of movies or TV Show released")
"""
observation:
1. United Kingdom is the only country among top 5 who released more TV shows than movies.
2. India released much lower TV Show in compare to movies.
"""

#let's make the area plot based on country and rating
country = data[['country', 'type - movie', 'type TV Show', 'rating']]
top_5_movie = country.groupby(['country','rating'], as_index = False).sum()
#top 5 countries who released most movies
top_5_movie.sort_values(['type - movie'], ascending = False, axis = 0, inplace = True)
top5 = top_5_movie.head(5)
top5.set_index("country", inplace = True)
print(top5)
top5.plot(kind = 'area', stacked = False, figsize = (20, 10))
plt.title("area plot of top 5 movie release and their ratings and countries")
plt.xlabel("ratings")
plt.ylabel("no. of movies or TV Show released")
"""
observation:
1. US release most of the movie with rating 'TV-MA', "R" and 'PG-13' but has not TV Show with rating 'R' and 'PG-13'
2. India release movie with rating 'TV-MA'
"""

data['release_year'].plot(kind = 'hist', figsize = (8,5))
#let's create bar graph for the top 5 countries movie- wise
bar_countries = data[['country', 'type - movie', 'type TV Show']]
bar_country = hist_countries.groupby(['country']).sum()
bar_country.sort_values(['type - movie'], ascending = False, inplace = True)
top = bar_country.head(5).transpose()
print(top)
top.plot(kind = "bar", figsize = (8, 5))

#Line chart
data1 = data[['release_year', 'type - movie', 'type TV Show']]
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
data2.set_index('release_year', inplace = True)
data2.plot(kind = 'line', figsize = (10, 6))
plt.title("line graph of movie and TV Show release")
plt.xlabel("number of movie and tv show release")
plt.show()
print(data2.tail())

import matplotlib as mpl
#this is the line graph of movie release by year
data1 = data[['release_year', 'type - movie']]
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
data2.set_index("release_year", inplace = True)
mpl.style.use('ggplot')
data2.plot(kind = "line", figsize = (8, 5))
plt.xlabel("years")
plt.ylabel("no. of movie released")
plt.title("movie released in the years between 1942 to 2021")
plt.show()


data3 = data2.iloc[-11:, :]
data3.reset_index(inplace = True)
data3.set_index('release_year', inplace = True)
data3.plot(kind = 'line', figsize = (8, 5))
plt.xlabel("years")
plt.ylabel("movie release")
plt.title("movie release in a decade from 2011 to 2021")
plt.text(2019, 600, 'covid hit')
plt.show()
#observation: release of movies peaked in the year 2017 and started decreasing agaain. most likely due to the covid.

#let's make a line graph for tv shows release.
data1 = data[['release_year', 'type TV Show']]
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
data2.set_index('release_year', inplace = True)
data2.plot(kind = 'line', figsize = (8, 5))
plt.xlabel("year")
plt.ylabel("TV Show release")
plt.title("TV Show release from 1942 to 2021")
plt.show()

#let's make graph from 2011 to 2021
data3 = data2.reset_index(inplace = True)
data3 = data2.iloc[-11:, :]
data3.set_index('release_year', inplace = True)
data3.plot(kind = 'line', figsize = (8,5))
plt.xlabel("years")
plt.ylabel("TV Show")
plt.title("TV Show release from 2011 to 2021")
plt.show()
#observation: TV Show peaked in the year 2020

#let's make combine graph for movie and TV show
data1 = data[['release_year', 'type - movie', 'type TV Show']]
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
data2.set_index('release_year', inplace = True)
data2.plot(kind = "line", figsize = (8,5))
plt.xlabel("years")
plt.ylabel("movie and TV show release")
plt.title("movie and TV show release from 1942 to 2021")

#let's make combine graph from 2011 to 2021
data3 = data2.iloc[-11:, :]
data3.reset_index(inplace = True)
data3.set_index('release_year', inplace = True)
data3.plot(kind = "line", figsize = (8, 5))
plt.xlabel("years")
plt.ylabel("TV Show and movie release")
plt.title("TV Show and movie release from 2011 to 2021")

#bar graph
data1 = data[['release_year', 'type - movie', 'type TV Show']]
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
data2.set_index('release_year', inplace = True)
data3 = data2.head(25)
data3.plot(kind = 'bar', figsize = (10, 6))
plt.title("bar graph of movie and TV Show release from 1942 to 1974")
plt.xlabel("years")
plt.ylabel("number of movie and tv show release")
plt.show()

data4 = data2.iloc[25:51, :]
data4.reset_index(inplace = True)
data4.set_index('release_year', inplace = True)
data4.plot(kind = 'bar', figsize = (10, 6))
plt.title("bar graph of movie and TV show release from 1975 to 2000")
plt.xlabel("years")
plt.ylabel("number of movie and tv show release")

data5 = data2.iloc[51:, :]
data5.reset_index(inplace = True)
data5.set_index('release_year', inplace = True)
data5.plot(kind = 'bar', figsize = (10, 6))
plt.title("bar graph of movie and TV show release from 2000 to 2021")
plt.xlabel("years")
plt.ylabel("number of movie and TV show release")

#observation: from 1942 to 1974, movie release was highest in the year 1973 and there was very less TV shows
#from 1975 to 2000, TV show release was highest in the year 1999 while movie release didn't change much
#from 2001, TV show release keeps on increasing so does movie release

data1 = data[['country', 'type - movie', 'type TV Show', 'release_year']]
data2 = data1.groupby(['country', 'release_year']).sum()
data2.reset_index(inplace = True)
#print(d)
#here, i want to compare movie and TV show release in India and USA. the first index is 0 where country is US
count = -1
#create an empty list to save the index of US and India
ans = []
#loop through the country column
for col in data2['country']:
    #if country is usa
    if col == "United States":
        #add 1 to count to get the correct index
        count += 1
        #save the index to our list
        ans.append(count)
    elif col == 'India':
        count +=1 
        ans.append(count)
    else:
        count += 1
#data3 contains only US and India country
data3 = data2.iloc[ans, :]

#print(data3[['country', 'type - movie', 'type TV Show']].transpose())
#print(data3)
data3.set_index('country', inplace = True)
data3.reset_index(inplace = True)
#print(data3)
#data4 = data3.groupby(['country'], inplace = True).sum()

"""data4 = data3.transpose()

data4.reset_index(drop = True,inplace = True)
#data.drop("index", axis =1, inplace = True)
#print(data4.columns)
#data4.set_index('release_year', inplace = True)
print(data4.head())"""
data3.set_index('release_year', inplace = True)
#data3.reset_index(inplace = True)
#data[['country']] = list(map(int, data['country']))
#print(data3)
#data3.columns = list(map(int, data.columns))
data3.plot(kind = 'area', figsize = (8, 5))

#scatterplot of movie release by year
import matplotlib.pyplot as plt
data1 = data[['release_year', 'type - movie']]
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
data2.plot(kind = 'scatter', x = 'release_year', y = "type - movie", figsize = (10, 6), color = 'darkblue')
plt.title("scatter plot of movie release based on years")
plt.xlabel("years from 1942 to 2021")
plt.ylabel("movie release")
#movie release has been consistent from the year 1942 to 2000
plt.show()

#scatter plot and regression line using seaborn
import seaborn as sns
width = 12
height = 10
plt.figure(figsize = (width, height))
sns.regplot(x = 'release_year', y = "type - movie", data = data2)
plt.title("movie release from 1942 to 2021")
plt.xlabel("release year")
plt.ylabel("number of movie")
plt.ylim(0,)
plt.show()

x = data2['release_year']
y = data2['type - movie']
fit = np.polyfit(x, y, deg = 1)
#plot the regression line on the scatterplot
data2.plot(kind = 'scatter', x = 'release_year', y = 'type - movie', figsize = (10, 6), color = 'darkblue')
plt.xlabel("years")
plt.ylabel("movie release")
plt.title("number of movie release from 1942 to 2021")
#plot line of best fit
plt.plot(x, fit[0]*x + fit[1], color = 'red')
plt.annotate('y = {0:.0f}x + {1:.0f}'.format(fit[0], fit[1]), xy = (1990, 400))
plt.show()

#let's create pie chart

data1 = data[['country', 'type - movie']]
data2 = data1.groupby(['country']).sum()
data2.sort_values(['type - movie'], ascending = False, inplace = True)
data2.head(25)
data2.reset_index(inplace = True)
data3 = data2[['country', 'type - movie']].head(6)
color_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0.1, 0, 0, 0.1]
data3['type - movie'].plot(kind = 'pie',
                          figsize = (15, 6),
                          autopct = '%1.1f%%',
                          startangle = 90,
                          shadow = True,
                          labels = None,
                          pctdistance = 1.12,
                          colors = color_list,
                          explode = explode_list)
plt.title("pie chart by country-wise", y = 1.16)
plt.axis('equal')
data3.reset_index(inplace = True)
data3.set_index('country', inplace = True)
plt.legend(labels = data3.index, loc = 'upper left')
plt.show()

data1 = data[['country', 'type TV Show']]
data2 = data1.groupby(['country']).sum()
data2.sort_values(['type TV Show'], ascending = False, inplace = True)
data2.reset_index(inplace = True)
data3 = data2[['country', 'type TV Show']].head(6)
color_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0.1, 0, 0, 0.1]
data3['type TV Show'].plot(kind = 'pie',
                          figsize = (15, 6),
                          autopct = '%1.1f%%',
                          startangle = 90,
                          shadow = True,
                          labels = None,
                          pctdistance = 1.12,
                          colors = color_list,
                          explode = explode_list)
plt.title("pie chart of TV show release by country", y = 1.16)
plt.axis('equal')
data3.reset_index(inplace = True)
data3.set_index('country', inplace = True)
plt.legend(labels = data3.index, loc = 'upper left')
plt.show()

#observation: US maintained first position
#we can say that TV show release has different audience than movie release

#let's make box plot for US
#first of all i will need all the rows with name 'United nation'
data1 = data[['country', 'release_year', 'type - movie']]
data2 = data1.groupby(['country', 'release_year']).sum()
data2.reset_index(inplace = True)
count = -1
country = []
for row in data2['country']:
    if row == 'United States':
        count += 1
        country.append(count)
    else:
        count += 1
data3 = data2.iloc[country, :]
data4 = data3.iloc[:, 1:]
data4.set_index('release_year', inplace = True)
#box plot: 1
data4.plot(kind = 'box', figsize = (8, 6), color = 'blue')
plt.title("box plot of movie release in US")
plt.ylabel("United States")
plt.xlabel("movie release")
plt.show()

#let's make box plot for US based on TV Show
data1 = data[['country', 'release_year', 'type TV Show']]
data2 = data1.groupby(['country', 'release_year']).sum()
data2.reset_index(inplace = True)
show = data2.iloc[country, 1:]
show.set_index("release_year", inplace = True)
show.plot(kind = 'box', figsize = (8, 6), color = 'blue')
plt.title("box plot of TV Show release in US")
plt.ylabel("United States")
plt.xlabel("TV show release")

#print(show)

#making bubble plot
data1 = data[['release_year', 'type - movie', 'type TV Show']]
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
#normalize the data
movie_norm = (data2['type - movie'] - data2['type - movie'].min())/(data2['type - movie'].max() - data2['type - movie'].min())
TV_norm = (data2['type TV Show'] - data2['type TV Show'].min())/(data2['type TV Show'].max() - data2['type TV Show'].min())
ax1 = data2.plot(kind = 'scatter',
                 x = 'release_year',
                 y = 'type - movie',
                 figsize = (14, 8),
                 alpha = 0.5,
                 color = 'green',
                 s = movie_norm * 2000 + 10,
                 xlim = (1942, 2021)
                )
ax2 = data2.plot(kind = 'scatter',
                 x = 'release_year',
                 y = 'type TV Show',
                 alpha = 0.5,
                 color = 'red',
                 s = TV_norm * 2000 + 10,
                 ax = ax1                 
                )
ax1.set_ylabel("number of movies and TV show release")
ax1.set_title("blubble plot of netflix release")
ax1.legend(['type - movie', 'type TV Show'], loc = 'upper left', fontsize = 'x-large')

import seaborn as sns
plt.figure(figsize = (15, 10))
sns.set(font_scale = 1.5)
ax = sns.regplot(x = 'release_year', y = 'type - movie', data = data2, color = 'green', marker = '+', scatter_kws={'s': 200})
ax.set(xlabel = 'years', ylabel = 'movie release')
ax.set_title("movie release by year")
plt.show()

plt.figure(figsize = (15, 10))
sns.set(font_scale = 1.5)
ax = sns.regplot(x = 'release_year', y = 'type TV Show', data = data2, color = 'red', marker = '+', scatter_kws ={'s' : 200})
ax.set(xlabel = 'years', ylabel = 'TV Show release')
ax.set_title("TV Show release by year")
plt.show() 

#wordcloud of the title of movies
from wordcloud import WordCloud, STOPWORDS
import urllib
stop = set(STOPWORDS)
word = WordCloud(background_color = 'white',max_words = 500,stopwords = stop)
s = ""
for title in data['title']:
    s = s + str(title)
fig = plt.figure(figsize = (18, 18))
word.generate(s)
plt.imshow(word, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#basic plotly chart
import plotly.express as px
import plotly.graph_objects as go
data1 = data.sample(n = 500, random_state = 42)
data1.shape
#how movie release changes w.r.t countries
#top 10 countries in term of movie release
data1 = data[['country', 'type - movie', 'type TV Show']]
data2 = data1.groupby(['country']).sum()
data2.sort_values(['type - movie'], ascending = False, inplace = True)
#let's take the top ten countries in term of movie release
data3 = data2.head(10)
data3.reset_index(inplace = True)
#how movie release changes w.r.t country
#creating a figure using go.figure and adding trace to it through go.scatter
fig = go.Figure(data = go.Scatter(x = data3['country'], y = data3['type - movie'], mode = 'markers', marker = dict(color = 'red')))
fig.update_layout(title = 'top 10 countries vs. movie release', xaxis_title = 'country', yaxis_title = "movie releaes")
fig.show()



#top 10 countries in term of TV Show release
data2.sort_values(['type TV Show'], ascending = False, inplace = True)
data3 = data2.head(10)
data3.reset_index(inplace = True)
fig = go.Figure(data = go.Scatter(x = data3['country'], y = data3['type TV Show'], mode = 'markers', marker = dict(color = 'blue')))
fig.update_layout(title = "top 10 countries in TV Show release", xaxis_title = 'country', yaxis_title = 'TV Show release')
fig.show()

#let's make a line plot of movie release from 1942 to 2021
line = data[['release_year', 'type - movie', 'type TV Show']]
line_Data = line.groupby(['release_year']).sum()
line_Data.reset_index(inplace = True)
fig = go.Figure(data = go.Scatter(x = line_Data['release_year'], y = line_Data['type - movie'], mode = "lines",
                                 marker = dict(color = 'green')))
fig.update_layout(title = "movie release from 1942 to 2021", xaxis_title = 'years', yaxis_title = 'movie release')
fig.show()

fig = go.Figure(data = go.Scatter(x = line_Data['release_year'], y = line_Data['type TV Show'], mode = "lines",
                                  marker = dict(color = 'green')))
fig.update_layout(title = "TV Show release from 1942 to 2021", xaxis_title = "years", yaxis_title = 'TV Show release')
fig.show()

data1 = data[['type - movie', 'listed_in']]
data2 = data1.groupby(['listed_in']).sum()
data2.sort_values(['type - movie'], ascending = False, inplace = True)
data3 = data2.head(10)
data3.reset_index(inplace = True)
fig = px.bar(data3, x = "listed_in", y = 'type - movie', title = 'movie release by type')
fig.show()
#creating a scatter plot
fig = px.scatter(data3, x = 'listed_in', y = 'type - movie', size = 'type - movie',
                hover_name = 'listed_in', title = "movies by their type", size_max = 60)
fig.show()

data1 = data[['country', 'type - movie', 'type TV Show']]
data2 = data1.groupby(['country']).sum()
data2.sort_values(['type - movie'], ascending = False, inplace = True)
#let's take the top ten countries in term of movie release
data3 = data2.head(10)
data3.reset_index(inplace = True)
fig = px.bar(data3, x = 'country', y = 'type - movie', title = 'movie release by top 10 countries')
fig.show()

#pie chart
fig = px.pie(data3, values = 'type - movie', names = 'country', title = 'pie chart of top 10 countries')
fig.show()

#bar chart
data2.sort_values(['type TV Show'], ascending = False, inplace = True)
data3 = data2.head(10)
data3.reset_index(inplace = True)
fig = px.bar(data3, x = 'country', y = 'type TV Show', title = 'tv show release by top 10 countries')
fig.show()

#pie chart
fig = px.pie(data3, values = 'type TV Show', names = 'country', title = 'pie chart of top 10 countries')
fig.show()

#bar Graph
data1 = data[['duration', 'type - movie']]
data2 = data1.groupby(['duration']).sum()
data2.sort_values(['type - movie'], ascending = False, inplace = True)
data3 = data2.head(10)
data3.reset_index(inplace = True)
fig = px.bar(data3, x = 'duration', y = 'type - movie', title = 'movies by their durations')
fig.show()

#bar graph
data1 = data[['duration', 'type TV Show']]
data2 = data1.groupby(['duration']).sum()
data2.sort_values(['type TV Show'], ascending = False, inplace = True)
data3 = data2.head(10)
data3.reset_index(inplace = True)
fig = px.bar(data3, x = 'duration', y = 'type TV Show', title = 'TV Show by their duration')
fig.show()

#scatterplot
data1 = data[['release_year', 'type - movie', 'type TV Show']]
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
fig = px.scatter(data2, x = 'release_year', y = 'type - movie', size = 'type - movie',
                hover_name = 'release_year', title = 'movie release by year', size_max = 60)
fig.show()

#sunburst graph shows hierarcy level
data1 = data[['release_year', 'country', 'type - movie', 'type TV Show']]
data2 = data1.groupby(['release_year', 'country']).sum()
data2.reset_index(inplace = True)
fig = px.sunburst(data2, path = ['release_year', 'country'], values = 'type - movie')
fig.show()

#histogram
data1 = data[['release_year', 'type - movie']]
data1.reset_index(inplace = True)
fig = px.histogram(data1, x = 'release_year')
fig.show()
data2 = data1.groupby(['release_year']).sum()
data2.reset_index(inplace = True)
fig = px.histogram(data2, x = 'release_year')
fig.show()

