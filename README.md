# Introduction to your project:

This project will explore the features of air pollution from sensors with data provided by OpenAQ, a nonprofit organization providing access to air quality data. We will employ supervised and unsupervised learning techniques. We will also use regression models to predict the type of pollutants from given features. Finally, we will apply clustering algorithms to cluster locations with similar pollution levels.

# Figures

# Methods
 This section details the exploration results, preprocessing steps, and models chosen in the ordered they were executed. Each step is subdivided for clarity, the methodology was chosen to comprehensively understand and analyze the air pollution data in the United States and processing large-scale environmental data. Ensuring data integrity and extracting meaningful insights. This allowed us for robust data cleaning, transformation and analysis, facilitating informed decision-making based on the findings.
Data Exploration
### 1. Creating a DataFrame from CSV files on the cloud


We started by creating a Data Frame from CSV files that were located at a specified path. The schema was defined to ensure proper data types for each column This step was crucial for our research, since this allowed us to handle large datasets efficiently using PySpark.
```path = "/expanse/lustre/projects/uci150/cmerry/data/records/csv.gz/country=us"```

```from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, DecimalType```

```schema = StructType([
StructField("location_id", IntegerType(), True),
StructField("sensors_id", IntegerType(), True),
StructField("location", StringType(), True),
StructField("datetime", TimestampType(), True),
StructField("lat", DecimalType(precision=10, scale=6), True),
StructField("lon", DecimalType(precision=10, scale=6), True),
StructField("parameter", StringType(), True),
StructField("units", String, True)```
2.       Pickling  and Unpickling
Since we were having difficulties working with such a large dataset in our environment (EXPANSE) , we opted to create a pickling file. Pickling is a method of serializing and deserializing a Python object structure, which in our research is our DataFrame. This allowed us to save the DataFrame for later use, and we also opted to unpickle the DataFrame for further processing.
# Load pickled RDD
```pickledRDD = spark.sparkContext.pickleFile("/expanse/lustre/projects/uci150/ameek1/df_pkl_full")```
# Define schema again
```schema = StructType([
   StructField("location_id", IntegerType(), True),
   StructField("sensors_id", IntegerType(), True),
   StructField("location", StringType(), True),
   StructField("datetime", TimestampType(), True),
   StructField("lat", DecimalType(precision=10, scale=6), True),
   StructField("lon", DecimalType(precision=10, scale=6), True),
   StructField("parameter", StringType(), True),
   StructField("units", StringType(), True),
   StructField("value", DecimalType(precision=10, scale=6), True), 
```

])
### 2. Convert RDD to DataFrame

```
df = spark.createDataFrame(pickledRDD, schema)
df.show()
df.count() # Output: 26541241
```

### 3. Preprocessing
To ensure data quality, several preprocessing steps were undertaken. These steps included transforming the datatime column, checking for null values, and filtering out invalid entries (due to malfunction of the sensors). The main goal of this section was to make sure the data was clean and reliable for subsequent analysis and potential modeling.
· Datetime column was split into date and time for better understanding:

```
from pyspark.sql.functions import date_format, col df = df.withColumn('date', date_format(col('datetime'), 'yyyy-MM-dd')) \ .withColumn('time', date_format(col('datetime'), 'HH:mm:ss')) df.show()
```
· Checking for null values, helped us identify and removed rows to maintain data integrity. It is known that null values can lead to inaccurate analysis and insights.
```
from pyspark.sql.functions import isnan, when, count, col df.select(*(count(when(col(c).isNull(), c)).alias(c) for c indf.columns)).show() # Filter rows with any null value df = df.na.drop("any") df.show()
```
· Removed the rows where the value had entries less than zero, as negative values (since they are not valid). This step was crucial to maintain validity of the pollution measurement .
```
import pyspark.sql.functions as func df = df.filter(col("value") > 0) df.show() # Save the cleaned DataFramedf.rdd.saveAsPickleFile("df_pkl_parsed")
```
### 4. M1: Data Analysis with PySpark SQL
Using PySpark SQL allowed for efficient querying and analysis of the large dataset. This was chosen for its scalability and ability to handle complex queries on big data.
We started by creating a temporary table and preformed several SQL queries to explore the data, such as counting distinct parameters and examining sensor counts.  We were also able to identify entries exceeding certain pollution thresholds reported to be harmful for humans.
### 5. M2: Graphical Anlaysis
Graphing our analysis was crucial, this was conducted to visualize trends and patterns in the air pollution data. Visualization helped us understand the temporal and spatial distribution of the pollution levels.
### 6. Additional Analysis: Handling Categorical Data
Since we were working with multiple columns, we realized we had some categorical variables. We opted to convert this data into numerical ones using label encoding. This with the goal of preparing the data for future machine learning models. We were also able to summarize the distribution of numerical columns providing insights into data characteristics. We then calculated the skewness and kurtosis for these numerical variables, with the goal to understand the shape and characteristics of the data distribution.
This concluded our method section for the second Milestone, detailing the exploration, preprocessing, and analysis steps taken during this section. These methodologies were chosen to ensure comprehensive and accurate analysis of air pollution.

# Results Section 

# Discussion Section

# Conclusion
In this project, we predicted pollutant concentrations using various supervised and unsupervised machine learning techniques. Preliminary data exploration of the OpenAQ Air Quality dataset guided our model selection.
We demonstrated the application of K-Means clustering to group air pollution data specific to the United States, identifying locations with similar pollution profiles. The optimal number of clusters, determined to be five, was established using the Elbow Method and Silhouette Analysis. Notably, Cluster 0 contained a significantly larger number of locations (327,305) compared to Clusters 1 through 4, which had 28, 7,836, 6,764, and 2,636 locations, respectively. This suggests that most locations exhibit similar pollution levels, likely representing smaller areas with comparable pollution, while fewer locations with higher pollution may correspond to larger urban centers. Despite tuning our model, the Within Set Sum of Squared Errors (WSSSE) was 99,674,957, indicating room for improvement.
In addition to clustering, we employed Time Series analysis to forecast levels of six pollutants (PM2.5, PM10, NO2, SO2, CO, and O3) based on historical data. We pivoted the historical air quality data across US cities, calculating the mean pollutant values per day for each city. This created a time series dataset with daily averages for each pollutant. After exploring different Time Series forecasting models, including ARIMA (AutoRegressive Integrated Moving Average), SARIMA (Seasonal ARIMA), and Prophet. We ultimately chose Prophet.  We then generated forecasts for future pollutant levels. This provided insights into expected air quality trends, enabling us to predict potential pollution events and assess long-term air quality changes.
For future work, validating sensor quality is crucial to improve analysis accuracy. Inconsistencies such as unrealistic sensor readings and missing data underscore the need for better sensor calibration and data verification.

# Statement of Collaboration:
Each team member was an integral part of this project. Ann Meek completed the abstract and part of the preprocessing and data exploration sections with Luisa Jaime. For preprocessing, Ann created a DataFrame with a specified schema from the CSV files. This DataFrame was converted to an RDD and pickled so that it could be used later once the notebook was re-opened, avoiding the need to recreate the DataFrame. This method worked well for the second milestone but ended up taking more time during milestone 3. Consequently, Ryan Thomson upgraded this to use Parquet files for better efficiency. Ann removed the preliminary null values and values pertaining to pollutants that were less than or equal to zero. 

Ann and Luisa also visualized specific sensors over time at particular locations. Luisa specifically analyzed statistics of the numerical values, that ensured the robustness of the data. She also  implemented a  label encoding on  the categorical variables for machine learning purposes. Additionally, Luisa created an interactive plot of the US, displaying sensor, parameter, and location data.

