"""
# This file contains the code for preparing the data and analysis.
    A. Data Preparation:
        -> Read data from csv file
        -> Understand data information and statistics
        -> Check feature ranges
        -> Check for missing values
        -> Check for duplicate records
        -> Identify outliers
        -> Data type conversion
        -> Feature engineering
    B. Exploritory Data Analysis (EDA)
        -> Univeriate analysis
        -> Bivariate analysis 
        -> Advanced Feature Engineering
"""

#%% Import libraries and handle warning errors

#Supress warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
# %%
