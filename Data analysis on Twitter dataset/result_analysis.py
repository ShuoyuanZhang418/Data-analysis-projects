# Import all packages needed in the project
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

# Set data frame display option
pd.set_option('display.unicode.ambiguous_as_wide', True)

pd.set_option('display.unicode.east_asian_width', True)

pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 5000)

pd.set_option('display.width', 5000)

pd.set_option('display.max_colwidth', 5000)

# read result csv file
# result_jul = pd.read_csv('FIFA_jul_result.csv', lineterminator='\n')
# result_jul_groupby = pd.read_csv('FIFA_jul_groupby_result.csv', lineterminator='\n')
# result_jul_0 = pd.read_csv('FIFA_jul_result_00.csv', lineterminator='\n')
# result_jul_groupby_0 = pd.read_csv('FIFA_jul_groupby_result_00.csv', lineterminator='\n')
# result_jul = pd.read_csv('data/FIFA_jul_result.csv', lineterminator='\n')
result_jul_groupby_0 = pd.read_csv('data/FIFA_jul_groupby_result.csv', lineterminator='\n')
result_jul_0 = pd.read_csv('data/FIFA_jul_result.csv', lineterminator='\n')

result_jul_h = pd.read_csv('data/FIFA_jul_result_h.csv', lineterminator='\n')
date_created = result_jul_h["created_at"].str.replace(' +0000 2014', '').str.split(' ')
date_day = date_created.map(lambda x: x[3])
date_hour = date_day.str.split(':')
result_jul_h["created_at"] = date_hour.map(lambda x: x[0]).astype(int)
print(result_jul_h.head())
print('result_jul_h data: Length: {} rows.'.format(len(result_jul_h)))
print(len(result_jul_h[["created_at"]]))

result_jul_h.to_csv('data/result_jul_created_at.csv')

result_aug = pd.read_csv('data/FIFA_aug_result.csv', lineterminator='\n')
result_aug_groupby = pd.read_csv('data/FIFA_aug_groupby_result.csv', lineterminator='\n')

result_sep = pd.read_csv('data/FIFA_sep_result.csv', lineterminator='\n')
result_sep_groupby = pd.read_csv('data/FIFA_sep_groupby_result.csv', lineterminator='\n')

result_oct = pd.read_csv('data/FIFA_oct_result.csv', lineterminator='\n')
result_oct_groupby = pd.read_csv('data/FIFA_oct_groupby_result.csv', lineterminator='\n')

result_nov = pd.read_csv('data/FIFA_nov_result.csv', lineterminator='\n')
result_nov_groupby = pd.read_csv('data/FIFA_nov_groupby_result.csv', lineterminator='\n')

# Show data types of each column
print(result_jul_0.dtypes)
print(result_jul_groupby_0.dtypes)

# Check the number of NULL in each column
# We can observed that there are a huge number of NaN in price column
print(result_jul_0.isnull().sum())
print(result_jul_groupby_0.isnull().sum())

# Replace NaN with 0
result_jul_0.fillna(0, inplace=True)
result_jul_groupby_0.fillna(0, inplace=True)


# World Cup top 16 teams(Hot teams)
# Brazil, Chile, Colombia, Uruguay, Netherlands, Mexico, Costa rica, Greek,
# French, Nigeria, Germany, Algeria, Argentina, Swiss, Belgium, USA(United States)

# The different stages of the competition
# group, Group, 1/8 final, Quarter final, quarter final, Semifinals, semifinals, finals

def monthly_tweets_number():
    month = []
    record_num = []
    print('result_jul data: Length: {} rows.'.format(len(result_jul_0)))
    month.append('July')
    record_num.append(len(result_jul_0))

    print('result_aug data: Length: {} rows.'.format(len(result_aug)))
    month.append('August')
    record_num.append(len(result_aug))

    print('result_sep data: Length: {} rows.'.format(len(result_sep)))
    month.append('September')
    record_num.append(len(result_sep))

    print('result_oct data: Length: {} rows.'.format(len(result_oct)))
    month.append('October')
    record_num.append(len(result_oct))

    print('result_nov data: Length: {} rows.'.format(len(result_nov)))
    month.append('November')
    record_num.append(len(result_nov))

    d = {"month": month, "Total number of record": record_num}
    monthly_record_num = pd.DataFrame(d)
    print(monthly_record_num)
    monthly_record_num.to_csv('data/monthly_record_num.csv')
    # print(result_jul_0['text'])


def get_user_description(name):
    return result_jul_0['description'][(result_jul_0['name'] == name)]


def get_tweet_text(name):
    return result_jul_0['text'][(result_jul_0['name'] == name)]


def sort_by_column(dataframe, column):
    df = dataframe.sort_values(by=column)
    return df


def find_top_16_rate():
    # Next we want to see which team is the most popular team in the World Cup top 16 teams
    # Brazil, Chile, Colombia, Uruguay, Netherlands, Mexico, Costa rica, Greek,
    # French, Nigeria, Germany, Algeria, Argentina, Swiss, Belgium, USA(United States)
    top_16_all = ['Brazil', 'Chile', 'Colombia', 'Uruguay', 'Netherlands', 'Mexico', 'Costa rica', 'Greek', 'French',
                  'Nigeria', 'Germany', 'Algeria', 'Argentina', 'Swiss', 'Belgium', 'USA', 'United States', 'ブラジル',
                  'チリ', 'コロンビア', 'ウルグアイ', 'オランダ', 'メキシコ', 'コスタリカ', 'ギリシャ', 'フランス', 'ナイジェリア',
                  'ドイツ', 'アルジェリア', 'アルゼンチン', 'スイス', 'ベルギー', 'アメリカ', 'アメリカ', "Бразилия", "Чили",
                  "Колумбия", "уругвай", "Нидерланды", "Мексика", "Коста - Рика", "Греция", "французский", "Нигерия",
                  "Германия", "алжир", "Аргентина", "Швейцария", "Бельгия", "Соединенные Штаты Америки", 'Brasil',
                  'Chile', 'Colômbia', 'Uruguai', 'Países Baixos', 'México', 'Costa rica', 'grego', 'francês',
                  'Nigéria', 'Alemanha', 'Argélia', 'Argentina', 'Suíça', 'Bélgica', 'EUA', 'Estados Unidos',
                  'Brasilien', 'Chile', 'Kolumbien', 'Uruguay', 'Niederlande', 'Mexiko', 'Costa Rica', 'Griechisch',
                  'Französisch', 'Nigeria', 'Deutschland', 'Algerien', 'Argentinien', 'Schweiz', 'Belgien', 'USA',
                  'Vereinigte Staaten']
    top_16_en_in = ['Brazil', 'Chile', 'Colombia', 'Uruguay', 'Netherlands', 'Mexico', 'Costa rica', 'Greek', 'French',
                    'Nigeria', 'Germany', 'Algeria', 'Argentina', 'Swiss', 'Belgium', 'USA', 'United States']
    top_16_ja = ['ブラジル', 'チリ', 'コロンビア', 'ウルグアイ', 'オランダ', 'メキシコ', 'コスタリカ', 'ギリシャ', 'フランス',
                 'ナイジェリア', 'ドイツ', 'アルジェリア', 'アルゼンチン', 'スイス', 'ベルギー', 'アメリカ', 'アメリカ']
    top_16_ru = ["Бразилия", "Чили", "Колумбия", "уругвай", "Нидерланды", "Мексика", "Коста - Рика", "Греция",
                 "французский", "Нигерия", "Германия", "алжир", "Аргентина", "Швейцария", "Бельгия",
                 "Америка"]
    top_16_dict = [['Brazil', 'ブラジル', "Бразилия", 'Brasil', 'Brasilien'], ['Chile', 'チリ', "Чили"],
                   ['Colombia', 'コロンビア', "Колумбия", 'Colômbia', 'Kolumbien'],
                   ['Uruguay', 'ウルグアイ', "уругвай", 'Uruguai', 'Uruguay'],
                   ['Netherlands', 'オランダ', "Нидерланды", 'Países Baixos', 'Niederlande'],
                   ['Mexico', 'メキシコ', "Мексика", 'México', 'Mexiko'],
                   ['Costa rica', 'コスタリカ', "Коста - Рика", 'Costa rica', 'Costa Rica'],
                   ['Greek', 'ギリシャ', "Греция", 'grego', 'Griechisch'],
                   ['French', 'フランス', "французский", 'francês', 'Französisch'],
                   ['Nigeria', 'ナイジェリア', "Нигерия", 'Nigéria', 'Nigeria'],
                   ['Germany', 'ドイツ', "Германия", 'Alemanha', 'Deutschland'],
                   ['Algeria', 'アルジェリア', "алжир", 'Argélia', 'Algerien'],
                   ['Argentina', 'アルゼンチン', "Аргентина", 'Argentina', 'Argentinien'],
                   ['Swiss', 'スイス', "Швейцария", 'Swiss', 'Schweiz'],
                   ['Belgium', 'ベルギー', "Бельгия", 'Bélgica', 'Belgien'],
                   ['USA', 'United States', 'アメリカ', "Америка", 'EUA', 'Estados Unidos', 'Vereinigte Staaten']]

    length = len(result_jul_0)
    flag = [False] * length
    for i in top_16_all:
        flag = (flag | result_jul_0.text.str.contains(i))
    print('The number of tweets that contain World Cup top 16 teams: ',
          len(result_jul_0['name'][flag]))
    all_num = len(result_jul_0['name'][flag])
    # print(result_jul_0['text'][(result_jul_0.text.str.contains('Brazil'))])
    # print(len(result_jul_0['name'][(result_jul_0.text.str.contains('Brazil'))]))
    # We can see that some of the tweets text contain 'Brazil 2014 World Cup' but not contain Brazil as a team
    # So we need to first delete this kind of tweet and count again to see whether this situation is frequent
    result_copy_0 = copy.deepcopy(result_jul_0)
    result_copy_0 = result_copy_0.drop(result_copy_0
                                       [(result_copy_0.text.str.contains('Чемпионат мира по футболу в Бразилии'))
                                        | (result_copy_0.text.str.contains('Brazil World Cup'))
                                        | (result_copy_0.text.str.contains('ブラジルワールドカップ'))
                                        | (result_copy_0.text.str.contains('Campeonato do Mundo Brasil'))].index)
    print('Brazil: ', len(result_jul_0['name'][(result_jul_0.text.str.contains('Brazil'))
                                                | (result_jul_0.text.str.contains('ブラジル'))
                                                | (result_jul_0.text.str.contains("Бразилия"))
                                                | (result_jul_0.text.str.contains('Brasil'))]))
    print('Brazil: ', len(result_copy_0['name'][(result_copy_0.text.str.contains('Brazil'))
                                                | (result_copy_0.text.str.contains('ブラジル'))
                                                | (result_copy_0.text.str.contains("Бразилия"))
                                                | (result_copy_0.text.str.contains('Brasil'))]))
    # We can see that the number of this situation is very few, so we can just ignoring it.
    length = len(result_copy_0)
    flag = [False] * length
    top_16_team = []
    rate = []
    for team in top_16_dict:
        for i in team:
            flag = (flag | result_copy_0.text.str.contains(i))
        print('%s:' %team[0], len(result_copy_0['name'][flag]))
        top_16_team.append(team[0])
        rate.append((len(result_copy_0['name'][flag]) / all_num))
        flag = [False] * length
    d = {"team": top_16_team, "popular rate": rate}
    top_16_rate = pd.DataFrame(d)
    print(top_16_rate)
    top_16_rate.sort_values(by=["popular rate"], inplace=True, ascending=False)
    print(top_16_rate)
    top_16_rate.to_csv('data/top_16_rate.csv')


def find_top_competitions():
    # Now we know which team is the most popular team, then we want to find which competition is the most popular one
    competitions = [['Brazil', 'Chile'], ['Colombia', 'Uruguay'], ['French', 'Nigeria'], ['Germany', 'Algeria'],
                    ['Netherlands', 'Mexico'], ['Costa rica', 'Greek'], ['Argentina', 'Swiss'],
                    ['Belgium', 'United States'], ['Brazil', 'Colombia'], ['French', 'Germany'],
                    ['Netherlands', 'Costa rica'], ['Argentina', 'Belgium'], ['Brazil', 'Germany'],
                    ['Netherlands', 'Argentina'], ['Germany', 'Argentina'], ['Brazil', 'Netherlands']]
    length = len(result_jul_0)
    flag = [False] * length
    for competition in competitions:
        flag = (flag | (result_jul_0.text.str.contains(competition[0])
                        & result_jul_0.text.str.contains(competition[1])))
    print('The number of tweets that contain World Cup competitions information: ',
          len(result_jul_0['name'][flag]))
    all_num = len(result_jul_0['name'][flag])
    flag = [False] * length

    top_competitions = []
    c_rate = []
    for competition in competitions:
        flag = (flag | (result_jul_0.text.str.contains(competition[0])
                        & result_jul_0.text.str.contains(competition[1])))
        # print('%s:' %team[0], len(result_jul_0['name'][flag]))
        competition_c = competition[0] + ' VS ' + competition[1]
        top_competitions.append(competition_c)
        c_rate.append((len(result_jul_0['name'][flag]) / all_num))
        flag = [False] * length
    d = {"competitions": top_competitions, "popular rate": c_rate}
    top_competitions_rate = pd.DataFrame(d)
    top_competitions_rate.sort_values(by="popular rate", inplace=True, ascending=False)
    print(top_competitions_rate)
    top_competitions_rate.to_csv('data/top_competitions_rate.csv')


def find_privacy_preference():
    language = pd.DataFrame(result_jul_0['lang'].value_counts())
    language.rename(columns={'lang': 'count'}, inplace=True)
    # l_num = sum(language['count'][0:10])
    l_num = sum(language['count'])
    # df = {"country": language.index[0:10], "proportion": language['count'][0:10] / l_num}
    df = {"language": language.index, "proportion": language['count'] / l_num}
    language_proportion = pd.DataFrame(df)
    language_proportion.reset_index(drop=True, inplace=True)
    print(language_proportion)
    language_proportion.to_csv('data/language_proportion.csv')

    country = pd.DataFrame(result_jul_0['country'][result_jul_0['country'] != 0].value_counts())
    country.rename(columns={'country': 'count'}, inplace=True)
    c_num = sum(country['count'])
    # c_num = sum(country['count'][0:10])
    df = {"country": country.index, "proportion": country['count'] / c_num}
    # df = {"country": country.index[0:10], "proportion": country['count'][0:10] / c_num}
    country_proportion = pd.DataFrame(df)
    country_proportion.reset_index(drop=True, inplace=True)
    print(country_proportion)
    country_proportion.to_csv('data/country_proportion.csv')

    # Because many users did not open geo location permissions. Now we can calculate whether users in different
    # countries are more likely to open permissions(below or high compare to the average level)
    a = country_proportion['proportion'][(country_proportion['country'] == '日本')].values
    b = language_proportion['proportion'][language_proportion['language'] == 'ja'].values
    print("Japan: ", a / b)
    a = country_proportion['proportion'][(country_proportion['country'] == 'Indonesia')].values
    b = language_proportion['proportion'][language_proportion['language'] == 'in'].values
    print("Indonesia: ", a / b)
    a = sum(country_proportion['proportion'][(country_proportion['country'] == 'United States') |
                                             (country_proportion['country'] == 'United Kingdom') |
                                             (country_proportion['country'] == 'Canada') |
                                             (country_proportion['country'] == 'Singapore') |
                                             (country_proportion['country'] == 'Ireland') |
                                             (country_proportion['country'] == 'India') |
                                             (country_proportion['country'] == 'Australia') |
                                             (country_proportion['country'] == 'South Africa')].values)
    b = language_proportion['proportion'][language_proportion['language'] == 'en'].values
    print("English-speaking countries: ", a / b)
    a = country_proportion['proportion'][(country_proportion['country'] == 'Russia')].values
    b = language_proportion['proportion'][language_proportion['language'] == 'ru'].values
    print("Russia: ", a / b)
    a = country_proportion['proportion'][(country_proportion['country'] == 'Deutschland')].values
    b = language_proportion['proportion'][language_proportion['language'] == 'de'].values
    print("Germany: ", a / b)
    a = country_proportion['proportion'][(country_proportion['country'] == 'China')].values
    b = language_proportion['proportion'][language_proportion['language'] == 'zh'].values
    print("China: ", a / b)


def find_users_activity():
    # We want to find the most active users
    # Official users
    print(result_jul_groupby_0[(result_jul_groupby_0['verified'] == True)][0:10])
    official_account = result_jul_groupby_0[['name', 'description', 'count'
                                             ]][(result_jul_groupby_0['verified'] == True)][0:5]
    official_account.to_csv("data/official_account.csv")
    # Unofficial users
    print(result_jul_groupby_0[(result_jul_groupby_0['verified'] == False)][0:10])
    unofficial_account = result_jul_groupby_0[['name', 'description', 'count'
                                               ]][(result_jul_groupby_0['verified'] == False)][0:5]
    unofficial_account.to_csv("data/unofficial_account.csv")


def count_unique_value():
    # Count how many times each unique value occurs
    b = pd.DataFrame(result_jul_0['lang'].value_counts())
    print(b)
    # print(pd.DataFrame(result_jul_0['favorite_count'].value_counts()))
    # print(pd.DataFrame(result_jul_0['retweet_count'].value_counts()))
    # print(pd.DataFrame(result_jul_0['city_name'].value_counts()))
    # print(pd.DataFrame(result_jul_0['country'].value_counts()))
    print(pd.DataFrame(result_jul_0['verified'].value_counts()))

    # print(result_jul_0['text'][(result_jul_0.lang == 'in')])
    # print(get_user_description('You Naldo'))
    # print(get_tweet_text('You Naldo'))
    # print(sort_by_column(result_jul_0, 'favorite_count'))
    # print(group_by_column(result_jul_0, 'lang'))

    # List the unique values of the column
    # a = list(result_jul_0['lang'].unique())
    # Count how many different values the column has
    # print('The number of unique values in language column is', len(a))
    # print('unique values in language column: ', a)


if __name__ == '__main__':
    # Find the popular rate of the top 16 team
    find_top_16_rate()

    # Find the popular rate of different competitions
    find_top_competitions()

    # Find privacy preference in different countries
    find_privacy_preference()

    # Find the most active users
    find_users_activity()

    # Count the unique values for each column in the data frame
    count_unique_value()

    # Count the total number of tweets related to the World Cup each month
    monthly_tweets_number()

    print(result_jul_0[(result_jul_0['lang'] == 'de')])








