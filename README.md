# Big Data with Spark Project

This project will explore the features of air pollution from sensors with data provided by OpenAQ, a nonprofit organization providing access to air quality data. We will employ supervised and unsupervised learning techniques. We will also use regression models to predict the type of pollutants from given features. Finally, we will apply clustering algorithms to cluster locations with similar pollution levels.

## Data Exploration

During the Data Exploration process, we initially downloaded the pollution data from OpenAQ which was housed on AWS. The data was downloaded into compressed csv.gz files which needed to be decompressed into csv files. After this, a DataFrame was created with specified schema from the csv files. This DataFrame was converted to a RDD and pickled so that it could be used later once the notebook was re-opened, so that we would not have to wait for the DataFrame to be re-created.

After this, we looked at the DataFrame. Null values and values pertaining to the pollutant that were less than or equal to zero were removed. These could be due to a sensor malfunction, as it would not make sense to have less than zero concentration, and would thus be recognized as outliers which would skew our data analysis. This parsed data was also pickled to use more readily, although in Milestone 2 we discovered ways to speed up our data pipeline.

From there, a specific location for a specific pollutant was visualized over time to see if there were trends. At first glance, it seemed like for location 1940, PM25 levels were generally higher during the summer. We also looked at statistics of the numerical values.


## Milestone 3 - First Model

### Further Pre-processing

In this milestone, we continued to preprocess the data. Although at an earlier 
milestone we saved the data to a pickle file so we didn't have to process the OpenAQ gzip files again, this was still a very slow process. We realized also at this point that we were missing many of the US locations we wanted to look at. OpenAQ has a path on AWS called "location=us", but unfortunately, for whatever reason, this does not contain a full set of US locations.

Therefore, we wrote code to query OpenAQ's API to get a list of all US locations with reference-grade sensors. Using this information, we downloaded the missing locations from AWS. Then to speed up future data loading and processing, we converted the OpenAQ S3 data to Parquet files. This process significantly speed up the data pipeline, enabling us to query and explore the data much quicker.

Next, realized that some locations in our dataset had longitude and latitude values that were equal to each other, so we removed those locations from our data.

We further pre-processed our data by converting our pollutant values to standardized units, as initially the data had inconsistent units, such as Celcius versus Fahrenheit.

Then we calculated daily averages for all parameters, and then pivoted the data to split up the parameters into their own columns.

The data had longitude and latitude values, but no other geographical information. Knowing we wanted to explore models that would benefit from categorical geographical location, we reverse geocoded the coordinates to get City, State, and Country locations.

Despite directly querying OpenAQ's API for a list of US locations, and downloading only those locations, looking at our new geopraphical information, we could see that the data still contained locations outside the US. So these locations were filtered out.

AQI (Air Quality Index) is a standardized score metric that is used to categorize the quality of air. We utilized a python AQI library that can convert between pollutant values and an AQI score, which can be found here: https://pypi.org/project/python-aqi/. Using the AQI library we were able to calculate a AQI score for each time point of data, and then using standardized thresholds for AQI scores, we categorized each AQI value by "good", "moderate", "unhealthy", etc. These thresholds for AQI can be found here: https://www.epa.gov/outdoor-air-quality-data/air-data-basic-information.


### K-Means Clustering

Finally we were able to look at out first model, for which we chose K-Means Clustering. Using this unsupervised model, our goal was to cluster the US air pollution data into clusters to investigate places that had similar pollution data.

First, we had to deal with the nulls in our dataset that were created when we averaged our data by day. We chose to replace the nulls with the average of the column. 

We then used the Elbow Method to plot the number of clusters, K, versus the within-cluster sum of squares, WSS, which measures the cluter compactness. 

We next performed a Silhouette Analysis, measuring the similarity of objects in their own clusters compared to other clusters. Higher scores mean the points are matched well to their own cluster.

The scores remained relatively stable until a K of 10, which it sharply dropped, indicating that at this value and beyond, there are too many clusters, leading to overfitting.

Through Elbow Method and Silhouette Analysis, we were able to determine that a good number of clusters would be five for our dataset. From the k = 5 clustering, we were able to see that, interestingly, the clusters that were created resulted in one large cluster, with 329764 locations, and 4 other smaller clusters. Finally, we calculated the error of our k means clustering.

### Next Model

For the next model, we would like to use the geographical information added via reverse geocoding. In particular, we're interested in using a classification model to predict air quality for different cities, or to compare geographical locations such as the East coast versus the West coast, or urban versus rural locations. 

### Conclusion

In conclusion, we can see that with the five clusters generated by our K-Means Clustering model, the first cluster contains many more locations than the other four. Cluster 0 contains 327,305 locations, while there are 28, 7836, 6764, and 2636 locations for Clusters 1 trough 4, respectively.

This interesting result sheds insight on how the majority of the air pollution data from locations around the US are all very similar, and hence why the majority of locations are clustered together. Perhaps this represents a large swath of smaller locations that exhibit similar pollution levels, compared to cities that deal with more pollution, such as larger urban centers. The answer would require further exploration and more data, like adding population data.

To evaluate how well our data was divided into these clusters, we calculated the Within Set Sum of Squared Errors which is a sum of squared distances for each data point to the centroid of its cluster. Even with our hyperparameter tuning through the elbow method and silhouette analysis above, we calculated this error to be 99674957, which is higher than we would have liked it to be. One way to improve it would have been to handle the missing data values better. Unfortunately, for our data set, many parameters were missing or null which led us to take the per-column average to replace these missing nulls. This could have skewed the results of this model.