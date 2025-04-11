"""
# This file contains the code for preparing the data and analysis.
    A. Data Preparation:
        -> Import Libraries 
        -> Read data from csv file
        -> Understand data information and statistics
        -> Check feature ranges
        -> Check for missing values
        -> Check for duplicate records
        -> Identify outliers
        -> Data type conversion
    B. Exploritory Data Analysis (EDA)
        -> Univeriate analysis
        -> Bivariate analysis 
        -> Feature Engineering
"""

# =========================================== A. Data Preparation =========================================== #


# ============== Import libraries and handle warning errors ============== #

#Supress warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# ============== Read data from csv file ============== #

# DataFrame from csv file
raw_data = pd.read_csv('./data/Student_performance_data .csv')
raw_data.head()


# ============== Understand data information and statistics ============== #

# Information
raw_data.info()
'''
This provides information about the data such as:
    -> Number of rows (student records)
    -> Number of columns (features)
    -> Feature names and their data types

Answer:
RangeIndex: 2392 entries, 0 to 2391
Data columns (total 15 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   StudentID          2392 non-null   int64  
 1   Age                2392 non-null   int64  
 2   Gender             2392 non-null   int64  
 3   Ethnicity          2392 non-null   int64  
 4   ParentalEducation  2392 non-null   int64  
 5   StudyTimeWeekly    2392 non-null   float64
 6   Absences           2392 non-null   int64  
 7   Tutoring           2392 non-null   int64  
 8   ParentalSupport    2392 non-null   int64  
 9   Extracurricular    2392 non-null   int64  
 10  Sports             2392 non-null   int64  
 11  Music              2392 non-null   int64  
 12  Volunteering       2392 non-null   int64  
 13  GPA                2392 non-null   float64
 14  GradeClass         2392 non-null   float64
dtypes: float64(3), int64(12)
memory usage: 280.4 KB
'''

# Statistics
raw_data.describe().T #Transposing the data for a better display
'''
This provides statistics about the data such as:
    -> Mean and standard deviation
    -> Feature minimum and maximum values
    -> Quartile data

Answer:
                    count 	mean 	        std 	    min 	        25% 	        50% 	        75% 	        max
StudentID 	        2392.0 	2196.500000 	690.655244 	1001.000000 	1598.750000 	2196.500000 	2794.250000 	3392.000000
Age 	            2392.0 	16.468645 	    1.123798 	15.000000 	    15.000000 	    16.000000 	    17.000000 	    18.000000
Gender 	            2392.0 	0.510870 	    0.499986 	0.000000 	    0.000000 	    1.000000 	    1.000000 	    1.000000
Ethnicity 	        2392.0 	0.877508 	    1.028476 	0.000000 	    0.000000 	    0.000000 	    2.000000 	    3.000000
ParentalEducation 	2392.0 	1.746237 	    1.000411 	0.000000 	    1.000000 	    2.000000 	    2.000000 	    4.000000
StudyTimeWeekly 	2392.0 	9.771992 	    5.652774 	0.001057 	    5.043079 	    9.705363 	    14.408410 	    19.978094
Absences 	        2392.0 	14.541388 	    8.467417 	0.000000 	    7.000000 	    15.000000 	    22.000000 	    29.000000
Tutoring 	        2392.0 	0.301421 	    0.458971 	0.000000 	    0.000000 	    0.000000 	    1.000000 	    1.000000
ParentalSupport 	2392.0 	2.122074 	    1.122813 	0.000000 	    1.000000 	    2.000000 	    3.000000 	    4.000000
Extracurricular 	2392.0 	0.383361 	    0.486307 	0.000000 	    0.000000 	    0.000000 	    1.000000 	    1.000000
Sports 	            2392.0 	0.303512 	    0.459870 	0.000000 	    0.000000 	    0.000000 	    1.000000 	    1.000000
Music 	            2392.0 	0.196906 	    0.397744 	0.000000 	    0.000000 	    0.000000 	    0.000000 	    1.000000
Volunteering 	    2392.0 	0.157191 	    0.364057 	0.000000 	    0.000000 	    0.000000 	    0.000000 	    1.000000
GPA 	            2392.0 	1.906186 	    0.915156 	0.000000 	    1.174803 	    1.893393 	    2.622216 	    4.000000
GradeClass 	        2392.0 	2.983696 	    1.233908 	0.000000 	    2.000000 	    4.000000 	    4.000000 	    4.000000
'''

# Dimensionality (rows, columns)
print(f'The data has {raw_data.shape[0]} rows and {raw_data.shape[1]} columns')
'''
Answer:

The data has 2392 rows and 15 columns
'''

# ============== Check if values of each feature are in the valid Range ============== #

# Check if StudentID values are between 1001 and 3392 (inclusive)
in_range = raw_data['StudentID'].between(1001, 3392).all()
print(in_range)  
'''
Answer:
True

Inference:
We can see that all enteries match valid student ID's
'''

#Age range (between 15-18)
raw_data['Age'].value_counts()
'''
Answer:
Age
15    630
16    593
17    587
18    582

Inference:
We can see that students are of an age between 15-18 and there are no invalid ages
'''

#Gender range (male= 0 , female = 1)
raw_data['Gender'].value_counts()
'''
Answer:
Gender
1    1222
0    1170

Inference:
We can see that no entries have an invalid gender category value
'''

#Ethnicity range (caucasian = 0, african american = 1, asian = 2, other = 3)
raw_data['Ethnicity'].value_counts()
'''
Aswer:
Ethnicity
0    1207
1     493
2     470
3     222

Inference:
We can see that no entries have an invalid ethnicity category value
'''

#Parental Education range (none = 0, high school = 1, some college =2, bachelor's = 3, higher study = 4)
raw_data['ParentalEducation'].value_counts()
'''
Aswer:
ParentalEducation
2    934
1    728
3    367
0    243
4    120

Inference:
We can see that no entries have an invalid ParentalEducation category value
'''

# Check if StudyTimeWeekly values are between 0 and 20 (inclusive)
in_range = raw_data['StudyTimeWeekly'].between(0, 20).all()
print(in_range)
'''
Aswer:
True

Inference:
We can see that all entries for StudyTimeWeekly fall in the valid range
'''

# Check if Absences values are between 0 and 30 (inclusive)
in_range = raw_data['Absences'].between(0, 30).all()
print(in_range)
'''
Aswer:
True

Inference:
We can see that all entries for Absences fall in the valid range
'''

#Tutoring range (no = 0, yes = 1)
raw_data['Tutoring'].value_counts()
'''
Aswer:
Tutoring
0    1671
1     721

Inference:
We can see that no entries have an invalid Tutoring category value
'''

#Parental Support range (none = 0, low = 1, moderate =2, high = 3, very high = 4)
raw_data['ParentalSupport'].value_counts()
'''
Aswer:
ParentalSupport
2    740
3    697
1    489
4    254
0    212

Inference:
We can see that no entries have an invalid ParentalSupport category value
'''

#Extracurricular range (no = 0, yes = 1)
raw_data['Extracurricular'].value_counts()
'''
Aswer:
Extracurricular
0    1475
1     917

Inference:
We can see that no entries have an invalid Extracurricular category value
'''

#Sports range (no = 0, yes = 1)
raw_data['Sports'].value_counts()
'''
Aswer:
Sports
0    1666
1     726

Inference:
We can see that no entries have an invalid Sports category value
'''

#Music range (no = 0, yes = 1)
raw_data['Music'].value_counts()
'''
Aswer:
Music
0    1921
1     471

Inference:
We can see that no entries have an invalid Music category value
'''

#Volunteering range (no = 0, yes = 1)
raw_data['Volunteering'].value_counts()
'''
Aswer:
Volunteering
0    2016
1     376

Inference:
We can see that no entries have an invalid Volunteering category value
'''

# Check if GPA values are between 0.0 and 4.0 (inclusive)
in_range = raw_data['GPA'].between(0.0, 4.0).all()
print(in_range)
'''
Aswer:
True

Inference:
We can see that all entries for GPA fall in the valid range
'''

# Identify the number of GPA values not in the average school GPA range 
in_range = raw_data['GPA'].between(2.0, 4.0).value_counts()
print(in_range)
'''
Aswer:
GPA
False    1274
True     1118

Inference:
We can start to identify which students would be classified as at risk.
Since 1274 students fall below the average GPA they may require targeted intervention
'''

#GradeClass range (A = 0, B = 1, C = 2, D = 3 F = 4)
raw_data['GradeClass'].value_counts()
'''
Aswer:
GradeClass
4.0    1211
3.0     414
2.0     391
1.0     269
0.0     107

Inference:
From the GradeClass we can see the number of students that fall into each category.
The students that recieve a class of F (0.0) and D (1.0) 
could help us identify the failing and at-risk of failing students
'''

# ============== Check for missing values ============== #

#Calculates what percentage of the total enteries have missing values
missing_values = (
    raw_data.isnull().sum()/len(raw_data)*100
).astype(int)

#Title and underline
print(f'Column\t\t\t% missing')
print(f'{"-"}'*35)

missing_values
'''
Answer:
Column			% missing
-----------------------------------

StudentID            0
Age                  0
Gender               0
Ethnicity            0
ParentalEducation    0
StudyTimeWeekly      0
Absences             0
Tutoring             0
ParentalSupport      0
Extracurricular      0
Sports               0
Music                0
Volunteering         0
GPA                  0
GradeClass           0

Inference:
This data shows that none of the features have missing values
'''

# ============== Check for duplicate data ============== #

#Check duplicated values
raw_data.duplicated().value_counts()
'''
Answer:
False    2392

Inference:
No student entries have been duplicated
'''

# ============== Check for outliers ============== #

# Make use of Boxplots to identify if any of the numerical features have outliers

# Boxplot for Age
fig = px.box(
    data_frame = raw_data,
    x='Age',
    orientation='h',
    title='Boxplot of Age Range',
)
fig.show()

# Boxplot for StudeyTimeWeekly
fig = px.box(
    data_frame = raw_data,
    x='StudyTimeWeekly',
    orientation='h',
    title='Boxplot of StudyTimeWeekly Range',
)
fig.show()

# Boxplot for Absences
fig = px.box(
    data_frame = raw_data,
    x='Absences',
    orientation='h',
    title='Boxplot of Absences Range',
)
fig.show()

# Boxplot for GPA
fig = px.box(
    data_frame = raw_data,
    x='GPA',
    orientation='h',
    title='Boxplot of GPA Range',
)
fig.show()

'''
Inference:
All values for each of the boxplots fell in an acceptable range 
and therefore no outliers were identified or removed
'''

# ============== Data Type Conversion ============== #

#Converts the GradeClass values into integres to better represent category values (A = 0, B=1, C=2, D=3, F=4)
raw_data['GradeClass'] = raw_data['GradeClass'].astype(int)
raw_data

# ============== Feature Engineering ============== #

# == Remove Irrelavent Features == #

#Drop irrelavent features (StudentID)
raw_data2 = raw_data.drop(columns=['StudentID'], inplace=False)

#Check dimensions
print(f'The data has {raw_data2.shape[0]} rows and {raw_data2.shape[1]} columns')
'''
StudentID will have no affect on their GPA or GradeClass and therefore was removed
We can check that the column was removed by looking at the shape of the dataset

Answer:
The data has 2392 rows and 14 columns (originally had 15 columns)
'''

# == Combine Features == #

#ExtracuricularEngagement
'''
This features allows us to combine multiple columns that are all related to extracuricular engagement
'''
raw_data2['ExtracurricularScore'] = (
    raw_data2['Extracurricular'] + 
    raw_data2['Sports'] + 
    raw_data2['Music'] + 
    raw_data2['Volunteering']
)

#Drop the individual columns and store them in raw_data3
raw_data3 = raw_data2.drop(columns=['Extracurricular', 'Sports', 'Music', 'Volunteering'])
print(f'The data has {raw_data3.shape[0]} rows and {raw_data3.shape[1]} columns')
raw_data3.head()
'''
Answer:
The data has 2392 rows and 11 columns

    Age 	Gender 	Ethnicity 	ParentalEducation 	StudyTimeWeekly 	Absences 	Tutoring 	ParentalSupport 	GPA 	    GradeClass 	ExtracurricularScore
0 	17 	    1 	    0 	        2 	                19.833723 	        7 	        1 	        2 	                2.929196 	2 	        1
1 	18 	    0 	    0 	        1 	                15.408756 	        0 	        0 	        1 	                3.042915 	1 	        0
2 	15 	    0 	    2 	        3 	                4.210570 	        26 	        0 	        2 	                0.112602 	4 	        0
3 	17 	    1 	    0 	        3 	                10.028829 	        14 	        0 	        3 	                2.054218 	3 	        1
4 	17 	    1 	    0 	        2 	                4.672495 	        17 	        1 	        3 	                1.288061 	4 	        0
'''


#StudeyAbsenceRatio
'''
The relationship between study time and absences could reveal valuable information about a student’s engagement. 
A student who spends a lot of time studying but has many absences may need extra support.
If ratio is high (>2) = lots of studying very little absence (dedicated student)
If ratio is low (<0.5) = little studying and very absent (potential at-risk students)
'''
#Make StudyAbsenceRatio from dividing StudyTimeWeekly by Absences taking into account zero divisibility
raw_data3['StudyAbsenceRatio'] = raw_data['StudyTimeWeekly'] / (raw_data['Absences'] + 1)
raw_data4 = raw_data3.copy()
raw_data4.head()
'''
Answer:

    Age 	Gender 	Ethnicity 	ParentalEducation 	StudyTimeWeekly 	Absences 	Tutoring 	ParentalSupport 	GPA 	    GradeClass 	ExtracurricularScore    StudyAbsenceRatio  
0 	17 	    1 	    0 	        2 	                19.833723 	        7 	        1 	        2 	                2.929196 	2 	        1                       2.479215
1 	18 	    0 	    0 	        1 	                15.408756 	        0 	        0 	        1 	                3.042915 	1 	        0                       15.408756
2 	15 	    0 	    2 	        3 	                4.210570 	        26 	        0 	        2 	                0.112602 	4 	        0                       0.155947
3 	17 	    1 	    0 	        3 	                10.028829 	        14 	        0 	        3 	                2.054218 	3 	        1                       0.668589
4 	17 	    1 	    0 	        2 	                4.672495 	        17 	        1 	        3 	                1.288061 	4 	        0                       0.259583
'''


#ParentalInfluenceScore
'''
Combining ParentalEducation, Parental Support and Tutoring provides an overall measure of academic support at home
Tutoring is given double the weight due to its direct impact on a student
'''
raw_data4["ParentalInfluenceScore"] = raw_data["ParentalSupport"] + raw_data["ParentalEducation"] + (raw_data["Tutoring"]*2)
raw_data5 = raw_data4.copy()
raw_data5.head()
'''
Answer:

    Age 	Gender 	Ethnicity 	ParentalEducation 	StudyTimeWeekly 	Absences 	Tutoring 	ParentalSupport 	GPA 	    GradeClass 	ExtracurricularScore    StudyAbsenceRatio   ParentalInfluenceScore
0 	17 	    1 	    0 	        2 	                19.833723 	        7 	        1 	        2 	                2.929196 	2 	        1                       2.479215            6
1 	18 	    0 	    0 	        1 	                15.408756 	        0 	        0 	        1 	                3.042915 	1 	        0                       15.408756           2
2 	15 	    0 	    2 	        3 	                4.210570 	        26 	        0 	        2 	                0.112602 	4 	        0                       0.155947            5
3 	17 	    1 	    0 	        3 	                10.028829 	        14 	        0 	        3 	                2.054218 	3 	        1                       0.668589            6
4 	17 	    1 	    0 	        2 	                4.672495 	        17 	        1 	        3 	                1.288061 	4 	        0                       0.259583            7
'''


# =========================================== B. Exploratory Data Analysis (EDA) =========================================== #

# ============== Univariate ============== #










# ============== Bivariate ============== #













# ============== Feature Engineering ============== #