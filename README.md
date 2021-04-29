 # Forecast Application

## Web app developed with Streamlit & Prophet

This application is a front end for forecasting using Prophet library. [Application link](https://share.streamlit.io/giandata/forecast-app/forecastapp.py)

The app structure follow the process of building and evaluating a forecast:

### 1. Load the time series

Upload a .csv file in form of a timeseries. Even though Prophet requires the date column o be labeled **ds** and the value column as **y** , it is not necessary to pre-proces the file to comply with this rule, since it is taken care of during the upload in the app. 
The dataset can as well contain multiple columns, of which only the chosen one in the selector will be used for the prediction.
After loading data a checkbox will show up to visualize the dataframe, include a statistical description and create a plot of the timeseries. 
### 2. Configure the model settings

Once loaded the data, the app allow the configuration of multiple parameters:
- **Horizon**: the time in future to forecast. It is expressed in days.
- **Seasonality**: choose beetwen Additive seasonality or Multiplicative seasonality. Useful when we can infer o have knoledge of the variation of volatility of the time series.
- **Trend components**: declare which trends want to discover and propagate. Daily should be selected if loading a dataset with hourly data.
Weekly: Prophet will search for thrend during days of the week (monday to sunday).
Monthly: Prophet will search for trend during days of the month (1th to 31th).
Yearly: will evaluate trend within months of the year( january to december)
-**Growth model**: choose beetwen linear growth or logistic growth, to specify  carrying capacity, for example if there is a maximum achievable point. The app allows then to specify cap and floor of the logistic model.
- **Holidays**: add holidays to the model. Available countries at the moment: Italy, Spain, France, United States, Germany, Ukraine.
- **Hyperparameters**: Change the scale of the changepoints or holidays. It impacts the flexibility of the model. 

### 3. Fit the model and predict future
- Initialize the model with the settings configured above  (Fit)
- Generate forecast (Predict): will plot forecast with standard Prophet charts
- Show components: shows finding about time components selected in point 2.

### 4. Evaluate and validate prediction

- Set the k-fold configuration: specify the initial timeframe to keep as training data, the horizon to predict and recurrency of the prediction as period.
- Calculate the metrics related to the cross-validatio. A dataframe will be generated and a plot of the selected metric will be created.

### 5. Hyperparameter tuning
Runs the model with all the combinations possible within the matrix of coefficients of scaling. It return the best combination of changepoint ans seasonality prior scale, which can be used to go back above at point 2 and embed in the model and create an optimized forecast.

### 6. Export results

- Export forecast(.csv) : will generate a link to download the dataframe with predictions and confidence intervals.
- Export model metrics (.csv): will generate a link to downloa d the dataframe or the cross-validation
- Export model configuration (.json): will export  the configuration of the model for reproducibility of the results.   

____________
### Author
[Giancarlo Di Donato](https://www.linkedin.com/in/giancarlodidonato/)


Last update: 29 April 2021