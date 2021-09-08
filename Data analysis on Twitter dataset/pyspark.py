# Import all packages needed in the project
from __future__ import print_function
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, DataFrame
from pyspark.sql.functions import desc, mean
from pyspark.sql.functions import array_contains
import pandas as pd

# Set up spark session using 'yarn' as master
conf = (SparkConf()
.setMaster("yarn") # if "local" then it will ran using the zone server's resources
.setAppName("s4607609-assignment-pyspark")) # you can choose any name that you want
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)


# Data exploration
df = sqlContext.read.json('/data/ProjectDatasetTwitter/statuses.log.2014-12-03.gz')

# clean df and select the columns are are interested in
df = df.na.drop(subset=["user.id"]).select(["user","entities", "lang", "retweeted", "favorited"])

# view the contents of a column
print(df.select('lang').show())
print(df.show())

# list the most common languages
print(df.groupby('lang').count().sort(desc('count')).show()) # about 5 minutes to run

# get a dataframe of the user data
user_df = df.select('user.*')
print(user_df.show())

# count of tweets per user
tweeters = user_df.groupby('screen_name').count()

# show how many times the most active users have tweeted
print(tweeters.sort(desc('count')).show()) # about 5 minutes to run

# show the mean number of tweets per person
print(tweeters.select(mean('count')).show()) # about 5 minutes to run

# get a dataframe of the users and the retweeted / favorited status
# This allows us to select values that are contained in nested dataframes
refined_df = df.select(['user.favourites_count', 'user.followers_count',
                        'user.friends_count', 'user.statuses_count', 'lang',
                        'retweeted', 'favorited'])
print(refined_df.describe().show()) # about 5 minutes to run

user_df_2 = df.select("user.id","user.screen_name").dropDuplicates()
tweeters_2 = user_df_2.groupBy("screen_name").agg({"id" : "count"})
tweeters_2.sort(desc("count(id)")).show()

tweeters_2.write.options(header='True', delimiter=',').csv("/user/s4607609/assignment/tweeters_2_result")

# Look at tweet volume over time for a certain hashtag
entity_df = df.select('entities.*')
print(entity_df.show())
print(entity_df.select('hashtags').show(truncate=False))

hashtag_df = entity_df.select('hashtags.text')
print(hashtag_df.show(truncate=False))
hashtags = hashtag_df.groupby('text').count()
print(hashtags.sort(desc('count')).show(truncate=False)) # about 5 minutes to run

hashtag_df.write.options(header='True', delimiter=',').csv("/user/s4607609/assignment/hashtag_df_result")
hashtag_df.printSchema()

print(hashtag_df.filter(array_contains(hashtag_df.text,"UQ")).show(truncate=False))

Marvels = hashtag_df.filter(array_contains(hashtag_df.text,"Marvel"))
print(Marvels.show(truncate=False))

MCUs = hashtag_df.filter(array_contains(hashtag_df.text,"MCU"))
print(MCUs.show(truncate=False))
print(MCUs.count())

# Focus on certain users (e.g., edu.au users and see which academics from which university are most active).
user_df.printSchema()

Movie_companies = user_df.filter((user_df.description.like("%Movie%")) & (user_df.verified  == True)).groupby('name').count()
print(Movie_companies.show(truncate=False))
print(Movie_companies.count())

Movie_companies_2 = user_df.filter(user_df.description.like("%Movie%")).groupby('name').count()
print(Movie_companies.show(truncate=False))


# Look at URLs included in tweets to understand which internet domains are most popular on Twitter.
URLs_df = entity_df.select('urls.url', 'urls.display_url', 'urls.expanded_url')
URLs_df.write.options(header='True', delimiter=',').csv("/user/s4607609/assignment/URLs_df_result")
URLs = URLs_df.groupby('expanded_url').count()
print(URLs.sort(desc('count')).show(truncate=False)) # about 5 minutes to run

# Look at a specific event or hashtag and look at who is tweeting about it.
event_df = df.select(['user.id', 'user.name', 'lang', 'entities.hashtags.text', 'entities.urls.expanded_url'])

Marvels_posters = event_df.select('id', 'name').filter(array_contains(event_df.text,"Marvel")).groupby('name').count()
print(Marvels_posters.show(truncate=False))
print(Marvels_posters.sort(desc('count')).show(truncate=False))
print(Marvels_posters.count())

event_df_2 = df.select(['user.id', 'user.name', 'user.url', 'user.description', 'user.verified', 'entities.hashtags.text'])
edu_au_verified = event_df_2.filter((event_df_2.url.like("%edu.au%")) & (event_df_2.verified  == True))
print(edu_au_verified.show(truncate=False))
print(edu_au_verified.groupby('name').count().show(truncate=False))

edu_au = event_df_2.filter((event_df_2.url.like("%edu.au%")))
print(edu_au.groupby('name').count().sort(desc('count')).show(truncate=False))
edu_verified = event_df_2.filter((event_df_2.url.like("%edu%")) & (event_df_2.verified  == True))
print(edu_au.groupby('name').count().sort(desc('count')).show(truncate=False))

edu = event_df_2.filter((event_df_2.url.like("%edu%")))
print(edu.groupby('name').count().sort(desc('count')).show(truncate=False))

# edu_au_2 = event_df_2.filter((event_df_2.url.like("%edu.au%")) | (event_df_2.url.like("%edu%") & event_df_2.url.like("%au%")))
edu_au_2 = event_df_2.filter((event_df_2.url.like("%edu.au%")) | (event_df_2.url.like("%.edu%") & event_df_2.url.like("%.au%")))
print(edu_au.count())
print(edu_au_2.count())

print(edu_au_2.groupby('name').count().sort(desc('count')).show(truncate=False))
print(edu_au_2.show(truncate=False))

# Data filtering
# all data in july
# all data in july
df_jul = sqlContext.read.json('/data/ProjectDatasetTwitter/statuses.log.2014-07-*.gz')
event_df_jul = df_jul.select(['created_at', 'user.id', 'user.name', 'user.url', 'user.description',
                              'user.verified', 'text', 'entities.urls.expanded_url',
                              'retweeted_status', 'retweet_count', 'favorite_count',
                              'lang', 'place.country', 'place.country_code', 'place.full_name'])
event_df_jul = event_df_jul.withColumnRenamed('full_name', 'city_name')
FIFA_jul = event_df_jul.select('created_at','id', 'name', 'description', 'verified', 'text',
                               'retweet_count', 'favorite_count', 'lang', 'country',
                               'country_code', 'city_name').filter((event_df_jul.text.like("%Brazil world cup%")
                                                                    | event_df_jul.text.like("%FIFA World Cup%")
                                                                    | event_df_jul.text.like("%world cup%")
                                                                    | event_df_jul.text.like("%World Cup%")
                                                                    | event_df_jul.text.like("%La coupe du monde%")
                                                                    | event_df_jul.text.like("%weltmeisterschaft%")
                                                                    | event_df_jul.text.like("%Чемпионат мира%")
                                                                    | event_df_jul.text.like("%ワールドカップ%")
                                                                    | event_df_jul.text.like("%世界杯%")
                                                                    | event_df_jul.text.like("%Copa do mundo%")))
FIFA_jul.toPandas().to_csv('FIFA_jul_result.csv')
FIFA_jul_groupby = FIFA_jul.groupby('name','description', 'verified', 'lang', 'country',
                                    'country_code', 'city_name').count().sort(desc('count'))
FIFA_jul_groupby.toPandas().to_csv('FIFA_jul_groupby_result.csv')


# all data in july
df_jul_h = sqlContext.read.json('/data/ProjectDatasetTwitter/statuses.log.2014-07-*.gz')
event_df_jul_h = df_jul_h.select(['created_at', 'user.id', 'user.name', 'user.url', 'user.description', 'user.verified',
                                  'text', 'entities.urls.expanded_url', 'retweeted_status', 'retweet_count',
                                  'favorite_count', 'lang', 'place.country', 'place.country_code', 'place.full_name',
                                  'entities.hashtags'])
event_df_jul_h = event_df_jul_h.withColumnRenamed('full_name', 'city_name')
FIFA_jul_h = event_df_jul_h.select('created_at','id', 'name', 'description', 'verified', 'text', 'retweet_count',
                                   'favorite_count', 'lang', 'country', 'country_code', 'city_name',
                                   'hashtags').filter((event_df_jul_h.text.like("%Brazil world cup%")
                                                                    | event_df_jul_h.text.like("%FIFA World Cup%") 
                                                                    | event_df_jul_h.text.like("%world cup%")
                                                                    | event_df_jul_h.text.like("%World Cup%")
                                                                    | event_df_jul_h.text.like("%La coupe du monde%") 
                                                                    | event_df_jul_h.text.like("%weltmeisterschaft%")
                                                                    | event_df_jul_h.text.like("%Чемпионат мира%")
                                                                    | event_df_jul_h.text.like("%ワールドカップ%")
                                                                    | event_df_jul_h.text.like("%世界杯%")
                                                                    | event_df_jul_h.text.like("%Copa do mundo%")))
FIFA_jul_h = FIFA_jul_h.select('created_at', 'id', 'name', 'description', 'verified', 'lang', 'country', 'country_code',
                               'city_name', 'hashtags.text')
FIFA_jul_h.toPandas().to_csv('FIFA_jul_result_h.csv')
FIFA_jul_groupby_h_1 = FIFA_jul_h.groupby('name', 'verified', 'text').count().sort(desc('count'))
FIFA_jul_groupby_h_1.toPandas().to_csv('FIFA_jul_groupby_result_h_1.csv')
FIFA_jul_groupby_h_2 = FIFA_jul_h.groupby('created_at').count().sort(desc('count'))
FIFA_jul_groupby_h_2.toPandas().to_csv('FIFA_jul_groupby_result_h_2.csv')


# all data in August
df_aug = sqlContext.read.json('/data/ProjectDatasetTwitter/statuses.log.2014-08-*.gz')
event_df_aug = df_aug.select(['user.id', 'user.name', 'user.url', 'user.description', 'user.verified', 'text',
                              'entities.urls.expanded_url', 'retweeted_status', 'retweet_count', 'favorite_count',
                              'lang', 'place.country', 'place.country_code', 'place.full_name'])
event_df_aug = event_df_aug.withColumnRenamed('full_name', 'city_name')
FIFA_aug = event_df_aug.select('id', 'name', 'description', 'verified', 'text', 'retweet_count', 'favorite_count',
                               'lang', 'country', 'country_code', 'city_name').filter((event_df_aug.text.like("%Brazil world cup%")
                                                                    | event_df_aug.text.like("%FIFA World Cup%") 
                                                                    | event_df_aug.text.like("%world cup%")
                                                                    | event_df_aug.text.like("%World Cup%")
                                                                    | event_df_aug.text.like("%La coupe du monde%") 
                                                                    | event_df_aug.text.like("%weltmeisterschaft%")
                                                                    | event_df_aug.text.like("%Чемпионат мира%")
                                                                    | event_df_aug.text.like("%ワールドカップ%")
                                                                    | event_df_aug.text.like("%世界杯%")
                                                                    | event_df_aug.text.like("%Copa do mundo%")))
FIFA_aug.toPandas().to_csv('FIFA_aug_result.csv')
FIFA_aug_groupby = FIFA_aug.groupby('name','description', 'verified', 'lang', 'country','country_code', 'city_name').count().sort(desc('count'))
FIFA_aug_groupby.toPandas().to_csv('FIFA_aug_groupby_result.csv')


# all data in September
df_sep = sqlContext.read.json('/data/ProjectDatasetTwitter/statuses.log.2014-09-*.gz')
event_df_sep = df_sep.select(['user.id', 'user.name', 'user.url', 'user.description', 'user.verified', 'text',
                              'entities.urls.expanded_url', 'retweeted_status', 'retweet_count', 'favorite_count',
                              'lang', 'place.country', 'place.country_code', 'place.full_name'])
event_df_sep = event_df_sep.withColumnRenamed('full_name', 'city_name')
FIFA_sep = event_df_sep.select('id', 'name', 'description', 'verified', 'text', 'retweet_count', 'favorite_count',
                               'lang', 'country', 'country_code', 'city_name').filter((event_df_sep.text.like("%Brazil world cup%")
                                                                    | event_df_sep.text.like("%FIFA World Cup%") 
                                                                    | event_df_sep.text.like("%world cup%")
                                                                    | event_df_sep.text.like("%World Cup%")
                                                                    | event_df_sep.text.like("%La coupe du monde%") 
                                                                    | event_df_sep.text.like("%weltmeisterschaft%")
                                                                    | event_df_sep.text.like("%Чемпионат мира%")
                                                                    | event_df_sep.text.like("%ワールドカップ%")
                                                                    | event_df_sep.text.like("%世界杯%")
                                                                    | event_df_sep.text.like("%Copa do mundo%")))
FIFA_sep.toPandas().to_csv('FIFA_sep_result.csv')
FIFA_sep_groupby = FIFA_sep.groupby('name','description', 'verified', 'lang', 'country', 'country_code', 'city_name').count().sort(desc('count'))
FIFA_sep_groupby.toPandas().to_csv('FIFA_sep_groupby_result.csv')


# all data in October
df_oct = sqlContext.read.json('/data/ProjectDatasetTwitter/statuses.log.2014-10-*.gz')
event_df_oct = df_oct.select(['user.id', 'user.name', 'user.url', 'user.description', 'user.verified', 'text',
                              'entities.urls.expanded_url', 'retweeted_status', 'retweet_count', 'favorite_count',
                              'lang', 'place.country', 'place.country_code', 'place.full_name'])
event_df_oct = event_df_oct.withColumnRenamed('full_name', 'city_name')
FIFA_oct = event_df_oct.select('id', 'name', 'description', 'verified', 'text', 'retweet_count', 'favorite_count',
                               'lang', 'country', 'country_code', 'city_name').filter((event_df_oct.text.like("%Brazil world cup%")
                                                                    | event_df_oct.text.like("%FIFA World Cup%") 
                                                                    | event_df_oct.text.like("%world cup%")
                                                                    | event_df_oct.text.like("%World Cup%")
                                                                    | event_df_oct.text.like("%La coupe du monde%") 
                                                                    | event_df_oct.text.like("%weltmeisterschaft%")
                                                                    | event_df_oct.text.like("%Чемпионат мира%")
                                                                    | event_df_oct.text.like("%ワールドカップ%")
                                                                    | event_df_oct.text.like("%世界杯%")
                                                                    | event_df_oct.text.like("%Copa do mundo%")))
FIFA_oct.toPandas().to_csv('FIFA_oct_result.csv')
FIFA_oct_groupby = FIFA_oct.groupby('name','description', 'verified', 'lang', 'country', 'country_code', 'city_name').count().sort(desc('count'))
FIFA_oct_groupby.toPandas().to_csv('FIFA_oct_groupby_result.csv')


# all data in November
df_nov = sqlContext.read.json('/data/ProjectDatasetTwitter/statuses.log.2014-11-*.gz')
event_df_nov = df_nov.select(['user.id', 'user.name', 'user.url', 'user.description', 'user.verified', 'text',
                              'entities.urls.expanded_url', 'retweeted_status', 'retweet_count', 'favorite_count',
                              'lang', 'place.country', 'place.country_code', 'place.full_name'])
event_df_nov = event_df_nov.withColumnRenamed('full_name', 'city_name')
FIFA_nov = event_df_nov.select('id', 'name', 'description', 'verified', 'text', 'retweet_count', 'favorite_count', 'lang', 'country',
                               'country_code', 'city_name').filter((event_df_nov.text.like("%Brazil world cup%")
                                                                    | event_df_nov.text.like("%FIFA World Cup%") 
                                                                    | event_df_nov.text.like("%world cup%")
                                                                    | event_df_nov.text.like("%World Cup%")
                                                                    | event_df_nov.text.like("%La coupe du monde%") 
                                                                    | event_df_nov.text.like("%weltmeisterschaft%")
                                                                    | event_df_nov.text.like("%Чемпионат мира%")
                                                                    | event_df_nov.text.like("%ワールドカップ%")
                                                                    | event_df_nov.text.like("%世界杯%")
                                                                    | event_df_nov.text.like("%Copa do mundo%")))
FIFA_nov.toPandas().to_csv('FIFA_nov_result.csv')
FIFA_nov_groupby = FIFA_nov.groupby('name','description', 'verified', 'lang', 'country', 'country_code', 'city_name').count().sort(desc('count'))
FIFA_nov_groupby.toPandas().to_csv('FIFA_nov_groupby_result.csv')
