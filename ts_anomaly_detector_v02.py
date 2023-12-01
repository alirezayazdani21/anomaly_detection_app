import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

###################################################################
###################################################################
# Load CSV file and prepare data
def load_data(file):
    df = pd.read_csv(file)
    return df

###################################################################

# Streamlit app
def main():
    st.title('Anomaly and Novelty Detection for Univariate Time Series\n Developed by: **Al Yazdani**')

    uploaded_file = st.file_uploader("Upload CSV file", type='csv')

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = df.bfill()

        st.write("Sample of the uploaded data:")
        st.write(df.head())
        

###################################################################

        date_column = st.selectbox('Select the date variable:', options=df.columns)
        df[date_column] = pd.to_datetime(df[date_column])
        
        unique_dates = df[date_column].values[:]
        min_date = min(unique_dates)
        max_date = max(unique_dates)
 
        value_column = st.selectbox('Select Value column', options=[col for col in df.columns if col != date_column])
        
        start_date, end_date = st.select_slider('Select the training start and end dates',
                                                options=unique_dates,
                                                value=(min_date,max_date)      
                                                )

        # Filter data based on selected date range
        df_short = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]


###################################################################

        model_choice = st.radio('Model', ['Z_Score','Robust_Z_Score','Quantile', 'Neighbors', 'Isolation_Forest'])

####################################################################

        # Anomaly detection based on chosen model
        if model_choice == 'Z_Score':
            st.subheader('Z_Score Model')
            # Selectbox for number of standard deviations
            std_devs = st.selectbox('Number of Standard Deviations', options=[3,2.5,2,1.5,1])
            # Calculate Z-score
            df_short['Z_score'] = (df_short[value_column] - df_short[value_column].mean()) / df_short[value_column].std()
            # Detect outliers
            outliers = df_short[(df_short[date_column] >= start_date) & (df_short[date_column] <= end_date) & (abs(df_short['Z_score']) > std_devs)]
            st.subheader('Novelty detection')
            # Predict outliers
            new_value_logic = st.radio('Do you have a new value to check?',['No','Yes'])
            if new_value_logic =='Yes':
                new_value = st.number_input('Please enter the new value and press enter')
                new_score = (new_value - df_short[value_column].mean()) / df_short[value_column].std()
                st.write("Is the new value an outlier?", new_score > std_devs)
                
            
#################################
        elif model_choice == 'Robust_Z_Score':
            st.subheader('Robust_Z_Score Model')
            
            # Selectbox for number of standard deviations
            std_devs = st.selectbox('Number of Standard Deviations', options=[3,2.5,2,1.5,1])
            # Calculate Z-score
            MAD = abs(df_short[value_column] - df_short[value_column].median()).median()
            df_short['Z_score'] = 0.6745 * (df_short[value_column] - df_short[value_column].median()) / MAD
            # Detect outliers
            outliers = df_short[(df_short[date_column] >= start_date) & (df_short[date_column] <= end_date) & (abs(df_short['Z_score']) > std_devs)]
            st.subheader('Novelty detection')
            # Predict outliers
            new_value_logic = st.radio('Do you have a new value to check?',['No','Yes'])
            if new_value_logic =='Yes':
                new_value = st.number_input('Please enter the new value and press enter')
                new_score = 0.6745 * (new_value - df_short[value_column].median()) / MAD
                st.write("Is the new value an outlier?", new_score > std_devs)


#################################
        elif model_choice == 'Quantile':
            st.subheader('Quantile Model')
            # Selectbox for lower and upper thresholds
            lower_threshold = st.selectbox('Lower Threshold', options=[0.01, 0.05, 0.03, 0.005, 0.003, 0, 0.1, 0.15])
            upper_threshold = st.selectbox('Upper Threshold', options=[0.95, 0.99, 0.97, 0.995, 0.997, 0.999, 1, 0.9, 0.85])
            # Calculate quantiles
            lower_quantile = df_short[value_column].quantile(lower_threshold)
            upper_quantile = df_short[value_column].quantile(upper_threshold)
            # Detect outliers
            outliers = df_short[(df_short[date_column] >= start_date) & (df_short[date_column] <= end_date) & ((df_short[value_column] < lower_quantile) | (df_short[value_column] > upper_quantile))]
            st.subheader('Novelty detection')
            # Predict outliers
            new_value_logic = st.radio('Do you have a new value to check?',['No','Yes'])
            if new_value_logic =='Yes':
                new_value = st.number_input('Please enter the new value and press enter')
                st.write("Is the new value an outlier?", (upper_quantile - new_value)*(new_value - lower_quantile) < 0)

#################################
        elif model_choice == 'Neighbors':
            st.subheader('Nearest Neighbors Model')
            contam = st.selectbox('Contamination proportion (suspected percentage of outliers )', options=[0.025, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3,'auto'])            
            neighbors = st.selectbox('Number of nearest neighbors', options=[5,3,2,7,10]) 
            clf = LocalOutlierFactor(n_neighbors=neighbors, contamination=contam).fit(df_short[[value_column]]) 
            df_short['outliers_binary'] = clf.fit_predict(df_short[[value_column]])
            outliers = df_short[(df_short[date_column] >= start_date) & (df_short[date_column] <= end_date) & (df_short['outliers_binary']==-1)]
            st.subheader('Novelty detection')
            new_value_logic = st.radio('Do you have a new value to check?',['No','Yes'])
            if new_value_logic =='Yes':
                new_value = st.number_input('Please enter the new value and press enter')
                new_clf = LocalOutlierFactor(n_neighbors=neighbors, contamination=contam,novelty=True).fit(df_short[[value_column]])
                new_score = new_clf.predict([[new_value]])[0]
                st.write("Is the new value an outlier?", new_score == -1)
                
#################################
        elif model_choice == 'Isolation_Forest':
            st.subheader('Isolation Forest Model')
            n_estim = st.selectbox('Number of estimators', options=[100, 50, 200, 300, 500])            
            clf = IsolationForest(n_estimators=n_estim,random_state=123).fit(df_short[[value_column]]) 
            df_short['outliers_binary'] = clf.fit_predict(df_short[[value_column]])
            outliers = df_short[(df_short[date_column] >= start_date) & (df_short[date_column] <= end_date) & (df_short['outliers_binary']==-1)]
            st.subheader('Novelty detection')
            new_value_logic = st.radio('Do you have a new value to check?',['No','Yes'])
            if new_value_logic =='Yes':
                new_value = st.number_input('Please enter the new value and press enter')
                new_clf = IsolationForest(n_estimators=n_estim,random_state=123).fit(df_short[[value_column]])
                new_score = new_clf.predict([[new_value]])[0]
                st.write("Is the new value an outlier?", new_score == -1)

###################################################################
###################################################################
###################################################################
        # Display detected outliers
        st.header('Results')
        
                # Summary statistics and Top 5 values displayed side by side
        col1, col2, col3= st.columns(3)

        # Outliers list
        with col1:
            st.subheader('Outliers:')
            st.write(outliers[[date_column, value_column]])

        # Outliers summary statistics
        with col2:
            st.subheader('Outliers Summary:')
            st.write(outliers[value_column].describe())
            
        # Non-outliers summary statistics
        non_outliers = df_short.drop(outliers.index)[value_column]
        with col3:
            st.subheader('Non-outliers Summary:')
            st.write(non_outliers.describe())
#################################

        # Plots
        fig = px.histogram(outliers[value_column])
        st.plotly_chart(fig, use_container_width=True)
        
#################################
        
        sns.set_theme(style='darkgrid')       
        fig, ax = plt.subplots() 
        ax.plot(df_short[date_column], df_short[value_column], label='Time Series')
        ax.scatter(outliers[date_column], outliers[value_column], label='Outliers', color='red')
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Highlighted outliers')
        plt.legend()
        st.pyplot(fig)
        
#################################


if __name__ == "__main__":
    main()
