# Introduction to your project:

For this project, our group worked with features of air pollution from sensors located all around the world provided by OpenAQ, a nonprofit organization providing access to air quality data. This dataset not only provided us with abundant timestamped data from different geographical regions all around the world, but also provided current and relevant information regarding the environment which we found very interesting to research further into. As we as a society become more aware and conscious about how we are affecting our environment, air quality and pollution data such as from this dataset become increasingly significant to look into as it can provide important insight. 

With the provided parameters given for each sensor location (o2, co2, co, humidity, pm1, pm25, pm10, pressure, etc.) along with the longitude and latitude of each sensor, there are several parameters at our disposal to create models for. For our first model, we decided to create a k-means clustering unsupervised model on the sensors located within the US. This provided us with a high level view to investigate how the air pollution data in cities across the US compared to each other and how we could group cities with similar pollution profiles. This clustering data also gave us insight into how many natural groups are formed from the wide variety of environmental data in our US dataset. Our second model looked into time series forecasting when trained on past time stamped data of our parameters. This gives us insight on our ability to predict future events from past data and the accuracy of our predictions from these models. 


# Figures

# Methods
 This section details the exploration results, preprocessing steps, and models chosen in the ordered they were executed. Each step is subdivided for clarity, the methodology was chosen to comprehensively understand and analyze the air pollution data in the United States and processing large-scale environmental data. Ensuring data integrity and extracting meaningful insights. This allowed us for robust data cleaning, transformation and analysis, facilitating informed decision-making based on the findings.
## Data Exploration
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

## 3. Preprocessing
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
Data Analysis with PySpark SQL
Using PySpark SQL allowed for efficient querying and analysis of the large dataset. This was chosen for its scalability and ability to handle complex queries on big data.
We started by creating a temporary table and preformed several SQL queries to explore the data, such as counting distinct parameters and examining sensor counts.  We were also able to identify entries exceeding certain pollution thresholds reported to be harmful for humans.

Graphical Anlaysis
Graphing our analysis was crucial, this was conducted to visualize trends and patterns in the air pollution data. Visualization helped us understand the temporal and spatial distribution of the pollution levels.

Additional Analysis: Handling Categorical Data
Since we were working with multiple columns, we realized we had some categorical variables. We opted to convert this data into numerical ones using label encoding. This with the goal of preparing the data for future machine learning models. We were also able to summarize the distribution of numerical columns providing insights into data characteristics. We then calculated the skewness and kurtosis for these numerical variables, with the goal to understand the shape and characteristics of the data distribution.
This concluded our method section for the second Milestone, detailing the exploration, preprocessing, and analysis steps taken during this section. These methodologies were chosen to ensure comprehensive and accurate analysis of air pollution.

## Model 1: K-means Clustering
We wrote code to query OpenAQ's API to get a list of all US locations with reference-grade sensors. Using this information, we downloaded the missing locations from AWS. Then to speed up future data loading and processing, we converted the OpenAQ S3 data to Parquet files. This process significantly speed up the data pipeline, enabling us to query and explore the data much quicker.

Next, realized that some locations in our dataset had longitude and latitude values that were equal to each other, so we removed those locations from our data.

We further pre-processed our data by converting our pollutant values to standardized units, as initially the data had inconsistent units, such as Celcius versus Fahrenheit.

Then we calculated daily averages for all parameters, and then pivoted the data to split up the parameters into their own columns.

The data had longitude and latitude values, but no other geographical information. Knowing we wanted to explore models that would benefit from categorical geographical location, we reverse geocoded the coordinates to get City, State, and Country locations.

Despite directly querying OpenAQ's API for a list of US locations, and downloading only those locations, looking at our new geopraphical information, we could see that the data still contained locations outside the US. So these locations were filtered out.

AQI (Air Quality Index) is a standardized score metric that is used to categorize the quality of air. We utilized a python AQI library that can convert between pollutant values and an AQI score, which can be found here: https://pypi.org/project/python-aqi/. Using the AQI library we were able to calculate a AQI score for each time point of data, and then using standardized thresholds for AQI scores, we categorized each AQI value by "good", "moderate", "unhealthy", etc. These thresholds for AQI can be found here: https://www.epa.gov/outdoor-air-quality-data/air-data-basic-information.

Finally we were able to look at out first model, for which we chose K-Means Clustering. Using this unsupervised model, our goal was to cluster the US air pollution data into clusters to investigate places that had similar pollution data.

First, we had to deal with the nulls in our dataset that were created when we averaged our data by day. We chose to replace the nulls with the average of the column.

We then used the Elbow Method to plot the number of clusters, K, versus the within-cluster sum of squares, WSS, which measures the cluter compactness.

We next performed a Silhouette Analysis, measuring the similarity of objects in their own clusters compared to other clusters. Higher scores mean the points are matched well to their own cluster.

## Model 2
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

Name: Shane Lin
Contribution: Shane Lin and Ryan Thomson worked together to code the first model (K-means clustering). Shane wrote code to pivot the data to split by parameter. Shane also wrote code to add an additional column to the dataframe that calculated the AQI score based on the parameter values and also another column that showed the category of AQI reading based on the calculated AQI score. Shane conducted the silhouette analysis and elbow method as hyperparameters to determine the best number of clusters to split the data into. Shane wrote the code for the K-means-clustering model and wrote the conclusion based on the results of the analysis. Shane also helped complete the writeup. 
