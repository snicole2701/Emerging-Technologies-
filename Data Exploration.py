import pandas as pd

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m1_survey_data.csv"
df = pd.read_csv(dataset_url)

#view the first five rows of the data to see the data layout
df.head()

#check the number of rows and columns in the dataset
print(len(df))
print(len(df.columns))

#identify the data type of each column. we may have to change some of them later to analyse the data
print(df.dtypes)

#what is the mean age of the survey participants
mean_age = df['Age'].mean()
print(mean_age)

#how many unique coutries had participants in the survey
unique_countries = df['Country'].nunique()
print(unique_countries)

#identify duplicate rows
duplicates = df[df.duplicated()]
print(duplicates)

#remove all duplicate rows
df.drop_duplicates(inplace=True)
df.to_csv('cleaned_data.csv', index=False)

#check number of rows to verify duplicates were dropped
print(len(df))
print(len(df.columns))

#find the missing values in all columns
missing_values = df.isnull().sum()
print(missing_values)

#how many missing values are there in the column 'WorkLoc'
missing_values_workloc = df['WorkLoc'].isnull().sum()
print(missing_values_workloc)

#value counts for column 'WorkLoc'
workloc_value_counts = df['WorkLoc'].value_counts()
print(workloc_value_counts)
#the value that is most frequent is 'Office'

#replace the missing values with 'Office'
df['WorkLoc'].fillna('Office', inplace = True)

#verify the missing values was replaced with 'Office' by checking that there are no longer any missing values int he column 'WorkLoc'
missing_values_workloc = df['WorkLoc'].isnull().sum()
print(missing_values_workloc)

#identify the different categories for column 'CompFreq'
unique_compfreq = df['CompFreq'].unique()
print(unique_compfreq)

#to normalize the information given in CompFreq we create a new column called 'NormalizedAnnualCompensation'
def normalize_compensation(row):
    if row['CompFreq'] == 'Yearly':
        return row['CompTotal']
    elif row['CompFreq'] == 'Monthly':
        return row['CompTotal'] * 12
    elif row['CompFreq'] == 'Weekly':
        return row['CompTotal'] * 52
df['NormalizedAnnualCompensation'] = df.apply(normalize_compensation, axis=1)

#install seaborn
#install matplotlib

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

#plot the distribution curve for the annual salaries in USD in column 'ConvertedComp'
sns.displot(df['ConvertedComp'], kde = True)
plt.title('Distrubution of Annual Salaries')
plt.xlabel('Annual Salary')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(rotation=90)
plt.tight_layout()

#view the distribution plot and save it in my project folder as a .png
plt.savefig('annual_salary_distribution_plot.png', bbox_inches='tight')
plt.show()

#plot a histogram for the annual salaries in USD in column 'ConvertedComp'
plt.hist(df['ConvertedComp'], bins=10, edgecolor='black')
plt.title('Histogram of Annual Compensation')
plt.xlabel('Annual Salary')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(rotation=90)
plt.tight_layout()

#view the histogram and save it in my project folder as a .png
plt.savefig('annual_salary_histogram.png', bbox_inches='tight')
plt.show()


#calculate the mediam of 'ConvertedComp'
converted_comp_median = df['ConvertedComp'].median()
print(converted_comp_median)

#how many participants identified themselves as 'Man'
count_man = df[df['Gender'] == 'Man'].shape[0]
print(count_man)

#what is the median ConvertedComp for participants that identified as 'Woman'
#filter dataset for participants identified as 'Woman'
woman_data = df[df['Gender'] == 'Woman']

#calculate the median
median_converted_comp_women = woman_data['ConvertedComp'].median()
print(median_converted_comp_women)

#statistical summary of participant age
age_summary = df['Age'].describe()
print(age_summary)

#plot a histogram for the column 'Age'
plt.figure()
plt.hist(df['Age'], bins=10, edgecolor='Red')
plt.title('Histogram of Age of Participants')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(rotation=90)
plt.tight_layout()

#view the histogram and save it in my project folder as a .png
plt.savefig('histogram_age.png', bbox_inches='tight')
plt.show()

#identify if their are outliers in 'ConvertedComp' column using box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['ConvertedComp'])
plt.title('Box Plot of Converted Compensation')
plt.xlabel('Converted Compensation')
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(rotation=90)
plt.tight_layout()

#view the hbox plot and save it in my project folder as a .png
plt.savefig('Coverted_Comp_box_plot.png', bbox_inches='tight')
plt.show()

#find the inter quartile range for 'ConvertedComp'
Q1 = df['ConvertedComp'].quantile(0.25)
Q3 = df['ConvertedComp'].quantile(0.75)

##calculate the interquartile range (IQR)
IQR = Q3 - Q1
print(IQR)

#calculate the upper and lower bounds of 'ConvertedComp'
lower_bound = Q1 - 1.5 * IQR
print(lower_bound)
upper_bound = Q3 + 1.5 * IQR
print(upper_bound)

#how many outliers are there in 'ConvertedComp'
outliers = df[(df['ConvertedComp'] < lower_bound) | (df['ConvertedComp'] > upper_bound)]
number_outliers = outliers.shape[0]
print(number_outliers)

#create a new dataframe in which the outliers are excluded
new_data = df[(df['ConvertedComp'] >= lower_bound) & (df['ConvertedComp'] <= upper_bound)]
new_data.to_csv('no_outliers_data.csv', index=False)

#find the correlation between age and other numerical data
numeric_data = new_data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
age_correlation = correlation_matrix['Age']
print(age_correlation)

#use SQL queries to extract data from RDBMS

import requests
import sqlite3
conn = sqlite3.connect("m4_survey_data.sqlite")
from sqlalchemy import create_engine

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m4_survey_data.sqlite'
response = requests.get(url)

if response.status_code == 200: 
    with open('m4_survey_data.sqlite', 'wb') as file: 
        file.write(response.content) 
    print("File downloaded successfully!") 
else: 
    print(f"Failed to download file. Status code: {response.status_code}")

#File downloaded successfully!

engine = create_engine('sqlite:///m4_survey_data.sqlite')

#verify what tables are in the database
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)
conn.close()

#plot a histogram of ConvertedComp.
query = "SELECT ConvertedComp FROM master"
data = pd.read_sql(query, engine)
data = data.dropna()

plt.figure(figsize=(10, 6))
plt.hist(data['ConvertedComp'], bins=30, edgecolor='black')
plt.title('Histogram of Converted Compensation')
plt.xlabel('Converted Compensation')
plt.ylabel('Frequency')

plt.ticklabel_format(style='plain', axis='x')
plt.savefig('sqldatabase_convertedcomp_histogram.png')
plt.show()

#plot a box plot for 'Age'
query = "SELECT Age FROM master"
data = pd.read_sql(query, engine)
data = data.dropna()
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Age'])
plt.title('Box Plot of Age')
plt.xlabel('Age')

plt.savefig('sqldatabase_age_boxplot.png')
plt.show()

#scatter plot of 'Age' and 'WorkWeekHrs'
query = "SELECT Age, WorkWeekHrs FROM master"
data = pd.read_sql(query, engine)
data = data.dropna()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['WorkWeekHrs'])
plt.title('Scatter Plot of Age vs. Work Week Hours')
plt.xlabel('Age')
plt.ylabel('Work Week Hours')

plt.savefig('sqldatabase_age_workweekhrs_scatter.png')
plt.show()

#Create a bubble plot of 'WorkWeekHrs' and 'CodeRevHrs', use 'Age' column as bubble size.
query = "SELECT Age, WorkWeekHrs, CodeRevHrs FROM master"
data = pd.read_sql(query, engine)
data = data.dropna()
norm = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
plt.figure(figsize=(12, 8))
bubble_plot = sns.scatterplot(
    x=data['WorkWeekHrs'],
    y=data['CodeRevHrs'],
    size=norm,
    sizes=(20, 200),
    hue=data['Age'],
    palette='viridis',
    legend=False,
    alpha=0.6
)
plt.title('Bubble Plot of Work Week Hours vs. Code Review Hours')
plt.xlabel('Work Week Hours')
plt.ylabel('Code Rev Hours')

plt.savefig('sqldatabase_workweekhrs_coderevhrs_bubble.png')
plt.show()

#Create a pie chart of the top 5 databases that respondents wish to learn next year. Label the pie chart with database names. Display percentages of each database on the pie chart.
query = "SELECT DatabaseDesireNextYear FROM DatabaseDesireNextYear"
data = pd.read_sql(query, engine)
data['DatabaseDesireNextYear'] = data['DatabaseDesireNextYear'].str.split(';')
all_databases = data.explode('DatabaseDesireNextYear')['DatabaseDesireNextYear'].value_counts()
top_5_databases = all_databases.head(5)
plt.clf()
plt.figure(figsize=(10, 7))
plt.pie(
    top_5_databases,
    labels=top_5_databases.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('viridis', len(top_5_databases))
)
plt.title('Top 5 Databases Respondents Wish to Learn Next Year')
plt.axis('equal')

plt.savefig('sqldatabase_desired_database_pie.png')
plt.show()

#stacked chart of median WorkWeekHrs and CodeRevHrs for the age group 30 to 35.
query = "SELECT Age, WorkWeekHrs, CodeRevHrs FROM master"
data = pd.read_sql(query, engine)
age_group = data[(data['Age'] >= 30) & (data['Age'] <= 35)]
median_workweek_hrs = age_group['WorkWeekHrs'].median()
median_coderev_hrs = age_group['CodeRevHrs'].median()
median_values = pd.DataFrame({
    'Category': ['WorkWeekHrs', 'CodeRevHrs'],
    'Median Hours': [median_workweek_hrs, median_coderev_hrs]
})

plt.clf()
plt.figure(figsize=(10, 6))
plt.bar(median_values['Category'], median_values['Median Hours'], color=['#1f77b4', '#ff7f0e'])
plt.title('Median WorkWeekHrs and CodeRevHrs for Age Group 30-35')
plt.xlabel('Category')
plt.ylabel('Median Hours')

plt.savefig('sqldatabase_workweekhrs_coderevhrs-median_stacked.png')
plt.show()

#plot the median ConvertedComp for all ages from 45 to 60
query = "SELECT Age, ConvertedComp FROM master"
data = pd.read_sql(query, engine)
age_filtered_data = data[(data['Age'] >= 45) & (data['Age'] <= 60)]
median_converted_comp = age_filtered_data.groupby('Age')['ConvertedComp'].median()

plt.clf()
plt.figure(figsize=(12, 8))
plt.plot(median_converted_comp.index, median_converted_comp.values, marker='o')
plt.title('Median Converted Compensation for Ages 45 to 60')
plt.xlabel('Age')
plt.ylabel('Median Converted Compensation')

plt.grid(True)

plt.savefig('sqldatabase_media_convertedcomp_line.png')
plt.show()

#horizontal bar chart using column MainBranch.
query = "SELECT MainBranch FROM master"
data = pd.read_sql(query, engine)
main_branch_counts = data['MainBranch'].value_counts()

plt.clf()
plt.figure(figsize=(12, 8))
main_branch_counts.plot(kind='barh', color='blue')
plt.title('Distribution of Main Branches')
plt.xlabel('Count')
plt.ylabel('Main Branch')

plt.tight_layout()

plt.savefig('sqldatabase_mainbranch_bar.png')
plt.show()

#how many respondents indicated that they currently work with 'SQL'? 
query = """
SELECT COUNT(*) as sql_users_count
FROM LanguageWorkedWith
WHERE LanguageWorkedWith LIKE '%SQL%'
"""
result = pd.read_sql(query, engine)
sql_users_count = result['sql_users_count'].iloc[0]
print(sql_users_count)

#in the list of most popular languages respondents wish to learn next year, what is the rank of Python?
query = "SELECT LanguageDesireNextYear FROM LanguageDesireNextYear"
data = pd.read_sql(query, engine)
data['LanguageDesireNextYear'] = data['LanguageDesireNextYear'].str.split(';')
all_languages = data.explode('LanguageDesireNextYear')['LanguageDesireNextYear'].value_counts()
python_rank = all_languages.reset_index().reset_index()
python_rank.columns = ['Rank', 'Language', 'Count']
python_rank = python_rank[python_rank['Language'] == 'Python'].iloc[0]['Rank'] + 1

print(python_rank)

