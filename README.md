The goal of this app is to enable the implementation of quick yet effective (univariate) time series anomaly and novelty detection algorithms. 


# Running the app

1- To run the "ts_anomaly_detector_v02.py", simply use in terminal "streamlit run ts_anomaly_detector_v02.py". 

2- This will open the app on: Local URL: http://localhost:8501 or Network URL: http://192.168.1.181:8501. 

3- Once the app is ready, you may upload a csv file (e.g. "anomaly_df.csv" using the browse files option. 
The data file should have a date column and a time series data column, herein referred to as "target". 

4- As soon as data is uploaded, the default settings kick in and a simple z-score model is run. 

5- The default parameters can be changed by, for example, changing the model type, using the slider to change start and end date for model training, changing thresholds and parameters of the anomaly detection algorithm (e.g. number of standard deviations, lower and upper quantiles, contamination percentage and number of nearest neighbors, number of estimators, etc). 

6- There is an option to choose if the user wants to predict novelty (i.e. predicting wether a new oncoming value is an outlier or not). Currently, the novelty prediction for only one new data point is enabled.  
 
7- The output from the models include a list of detected outliers and its summary statistics (and summary statistics for training data not flagged as outlier), a histogram of the outliers distribution, and a visualization of training time series where the detected outliers are highlighted in red. 

# Ways to improve 
There are several ways to improve this project, for example: enabling additional anomaly detection algorithms that are suitable for either univariate or multi-variate tme series, incorporating mechanisms to reduce the discrepancy among models and introducing metrics to enable monitoring the quality of the models. 

Thank you.
Al Yazdani


