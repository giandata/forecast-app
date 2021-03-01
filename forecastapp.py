import streamlit as st
import pandas as pd
import numpy as np

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import json
from fbprophet.serialize import model_to_json, model_from_json
import holidays

import altair as alt
import plotly as plt
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import base64
import itertools
from datetime import datetime

st.set_page_config(page_title ="Forecast App")


tabs = ["Application","About"]
page = st.sidebar.radio("Tabs",tabs)

@st.cache(persist=True,suppress_st_warning=True,show_spinner= True)
def load_csv():
    
    df_input = pd.DataFrame()   # fa il dataframe
    
    df_input=pd.read_csv(input,sep=None , encoding='utf-8',
                            parse_dates=True,
                            infer_datetime_format=True)
    return df_input

def prep_data(df):
# rinomina colonne
    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.markdown("The selected date column is now labeled as **ds** and the values columns as **y**")
    df_input = df_input[['ds','y']]
    # assicuro che sia datetime
    #df_input['ds'] = pd.to_datetime(df_input['ds'])
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input


    #df_input = df_input.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    
    
    # uso  solo la colonna 0 e 1
    #df_input = (df_input.iloc[:,[0,1]])

    # ordino per colonna 0



if page == "Application":


    st.title('Forecast application 🧙🏻')
    st.write('This app enables you to generate time series forecast withouth any dependencies.')
    st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")
    df =  pd.DataFrame()   

    st.subheader('1. Data loading 🏋️')

    with st.beta_expander("Data format"):
            st.write("Import a time series csv file. The dataset can contain multiple columns but you will need to select a column to be used as dates and a second column containing the metric you wish to forecast. The columns will be renamed as **ds** and **y** to be compliant with Prophet. Even though we are using the default Pandas date parser, the ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric.")



    input = st.file_uploader('Upload time series.')

    if input:
        with st.spinner('Loading data..'):
            df = load_csv()
            

            st.write("Columns:")
            st.write(list(df.columns))
    
            columns = list(df.columns)
    
            col1,col2 = st.beta_columns(2)
    
            with col1:
                date_col = st.selectbox("Select date column",index= 0,options=columns,key="date")
    
            with col2:
                metric_col = st.selectbox("Select values column",index=1,options=columns,key="values")

            df = prep_data(df)


        if st.checkbox('Show data',key='show'):
            with st.spinner('Plotting data..'):
        
                st.dataframe(df)
                st.write("Dataframe description:")

                st.write(df.describe())
                try:
                    line_chart = alt.Chart(df).mark_line().encode(
                        x = 'ds:T',
                        y = "y").properties(title="Time series preview").interactive()
                    st.altair_chart(line_chart,use_container_width=True)
                except:
                    st.line_chart(df['y'],use_container_width =True,height = 300)
            
                
    st.subheader("2. Parameters configuration 🛠️")

    with st.beta_container():
        st.write('In this section you can modify the algorithm settings.')
            
        with st.beta_expander("Horizon"):
            periods_input = st.number_input('Select how many future periods (days) to forecast.',
            min_value = 1, max_value = 366,value=90)

        with st.beta_expander("Seasonality"):
            st.write("The seasonality depends on the specific case, therefore specific domain knowledge is required.")
            seasonality = st.radio(label='Seasonality',options=['additive','multiplicative'])

        with st.beta_expander("Trend components"):
            st.write("Add or remove components:")
            daily = st.checkbox("Daily")
            weekly= st.checkbox("Weekly")
            monthly = st.checkbox("Monthly")
            yearly = st.checkbox("Yearly")

        with st.beta_expander("Growth model"):
            st.write('Prophet uses by default a linear growth model.')
            growth = st.radio(label='Growth model',options=['linear',"logistic"]) 

            if growth == 'linear':
                growth_settings= {
                            'cap':1,
                            'floor':0
                        }
                cap=1
                floor=1
                df['cap']=1
                df['floor']=0

            if growth == 'logistic':
                st.info('Configure saturation')
                cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
                floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
            if floor > cap:
                st.error('Invalid settings.Cap must be higher then floor.')
                growth_settings={}
            else:
                growth_settings = {'cap':cap,'floor':floor}
                df['cap']=cap
                df['floor']=floor

        with st.beta_expander('Holidays'):
            #st.markdown("""[Available countries list](https://github.com/dr-prodigy/python-holidays) """)
            
            countries = ['Country name','Italy','Spain','United States','France','Germany','Ukraine']
            
            with st.beta_container():
                years=[2021]
                selected_country = st.selectbox(label="Select country",options=countries)

                if selected_country == 'Italy':
                    for date, name in sorted(holidays.IT(years=years).items()):
                        st.write(date,name) 
                            
                if selected_country == 'Spain':
                    
                    for date, name in sorted(holidays.ES(years=years).items()):
                            st.write(date,name)                      

                if selected_country == 'United States':
                    
                    for date, name in sorted(holidays.US(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'France':
                    
                    for date, name in sorted(holidays.FR(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'Germany':
                    
                    for date, name in sorted(holidays.DE(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'Ukraine':
                    
                    for date, name in sorted(holidays.UKR(years=years).items()):
                            st.write(date,name)

                else:
                    holidays = False
                            
                holidays = st.checkbox('Add country holidays to the model')
                    

            #with col2:
                #years = list(range(2000,2021,1))
        
                #years= st.multiselect("Add years",options=years)
                #st.write(f'Adding years {years}')


        with st.beta_expander('Hyperparameters'):
            st.write('In this section it is possible to tune the scaling coefficients.')
                
            changepoint_scale_values= [0.01, 0.1, 0.5]
            seasonality_scale_values= [0.1, 1.0, 10.0]
            changepoint_scale= st.select_slider(label= 'Changepoint prior scale',options=changepoint_scale_values)
            seasonality_scale= st.select_slider(label= 'Seasonality prior scale',options=seasonality_scale_values)


    with st.beta_container():
        st.subheader("3. Forecast 🔮")
        st.write("Fit the model on the data and generate future prediction.")
        st.write("Load a time series to activate.")
        if input:
            
            if st.checkbox("Initialize model (Fit)",key="fit"):
                if len(growth_settings)==2:
                    m = Prophet(seasonality_mode=seasonality,
                                daily_seasonality=daily,
                                weekly_seasonality=weekly,
                                yearly_seasonality=yearly,
                                growth=growth,
                                changepoint_prior_scale=changepoint_scale,
                                seasonality_prior_scale= seasonality_scale)
                    if holidays:
                        m.add_country_holidays(country_name=selected_country)
                        
                    if monthly:
                        m.add_seasonality(name='monthly', period=30.4375, fourier_order=4)

                    with st.spinner('Fitting the model..'):

                        m = m.fit(df)
                        future = m.make_future_dataframe(periods=periods_input,freq='D')
                        future['cap']=cap
                        future['floor']=floor
                        st.write("The model will produce forecast up to ", future['ds'].max())
                        st.success('Model fitted successfully')

                else:
                    st.warning('Invalid configuration')

            if st.checkbox("Generate forecast (Predict)",key="predict"):
                try:
                    with st.spinner("Forecasting.."):

                        forecast = m.predict(future)
                        st.success('Prediction generated successfully')
                        st.dataframe(forecast)
                        fig1 = m.plot(forecast)
                        st.write(fig1)

                        if growth == 'linear':
                            fig2 = m.plot(forecast)
                            a = add_changepoints_to_plot(fig2.gca(), m, forecast)
                            st.write(fig2)
                except:
                    st.warning("You need to train the model first.. ")
                        
            
            if st.checkbox('Show components'):
                try:
                    with st.spinner("Loading.."):
                        fig3 = m.plot_components(forecast)
                        st.write(fig3)
                except:
                    st.warning("Requires forecast generation..") 

        st.subheader('4. Model validation 🧪')
        st.write("In this section it is possible to do cross-validation of the model.")
        with st.beta_expander("Explanation"):
            st.markdown("""The Prophet library makes it possible to divide our historical data into training data and testing data for cross validation. The main concepts for cross validation with Prophet are:""")
            st.write("Training data (initial): The amount of data set aside for training. The parameter is in the API called initial.")
            st.write("Horizon: The data set aside for validation.")
            st.write("Cutoff (period): a forecast is made for every observed point between cutoff and cutoff + horizon.""")

            
        with st.beta_expander("Cross validation"):    
            initial = st.number_input(value= 330,label="initial",min_value=30,max_value=1096)
            initial = str(initial) + " days"

            period = st.number_input(value= 180,label="period",min_value=1,max_value=365)
            period = str(period) + " days"

            horizon = st.number_input(value= 120, label="horizon",min_value=30,max_value=366)
            horizon = str(horizon) + " days"

            st.write(f"Here we do cross-validation to assess prediction performance on a horizon of **{horizon}** days, starting with **{initial}** days of training data in the first cutoff and then making predictions every **{period}** days.")
            
        with st.beta_expander("Metrics definition"):
                    st.write("Mse: mean absolute error")
                    st.write("Rmse: mean squared error")
                    st.write("Mae: Mean average error")
                    st.write("Mape: Mean average percentage error")
                    st.write("Mdape: Median average percentage error")
            
        with st.beta_expander("Metrics"):
            if st.checkbox('Calculate metrics'):
                with st.spinner("Cross validating.."):

                    df_cv = cross_validation(m, initial=initial,
                                                period=period, 
                                                horizon = horizon,
                                                parallel="threads")
                    #In Python, the string for initial, period, and horizon should be in the format used by Pandas Timedelta, which accepts units of days or shorter.
                    # custom cutoffs = pd.to_datetime(['2019-11-31', '2019-06-31', '2021-01-31'])
                    
                    df_p = performance_metrics(df_cv)
                    st.dataframe(df_p)
                    
                    metrics = ['mse','rmse','mae','mape','mdape','coverage']
                    
                    selected_metric = st.radio(label='Plot metric',options=metrics)
                    st.write(selected_metric)
                    fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
                    st.write(fig4)

        st.subheader('5. Hyperparameter Tuning 🧲')
        st.write("In this section it is possible to find the best combination of hyperparamenters.")

        param_grid = {  
                            'changepoint_prior_scale': [0.01, 0.1, 0.5],
                            'seasonality_prior_scale': [0.1, 1.0, 10.0],
                        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        if st.button("Optimize hyperparameters"):
            with st.spinner("Finiding best combination. Please wait.."):


                # Use cross validation to evaluate all parameters
                for params in all_params:
                    m = Prophet(**params).fit(df)  # Fit model with given params
                    df_cv = cross_validation(m, initial=initial,
                                                    period=period,
                                                    horizon=horizon,
                                                    parallel="threads")
                    df_p = performance_metrics(df_cv, rolling_window=1)
                    rmses.append(df_p['rmse'].values[0])

                # Find the best parameters
                tuning_results = pd.DataFrame(all_params)
                tuning_results['rmse'] = rmses
                st.write(tuning_results)
                    
                best_params = all_params[np.argmin(rmses)]
                st.write(f'The best parameter tuning is {best_params}')

        st.subheader('6. Export results ✨')
            
        col1, col2, col3 = st.beta_columns(3)

        with col1:
            if st.button('Export forecast (.csv)'):
                with st.spinner("Exporting..",key="csv"):

                    export_forecast = pd.DataFrame(forecast[['ds','yhat_lower','yhat','yhat_upper']]).to_csv()
                    b64 = base64.b64encode(export_forecast.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click derecho > guardar como **forecast.csv**)'
                    st.markdown(href, unsafe_allow_html=True)
            
        with col2:
            if st.button("Export model metrics (.csv)"):
                with st.spinner("Exporting..",key="metrics"):
                    b64 = base64.b64encode(df_p.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click derecho > guardar como **metrics.csv**)'
                    st.markdown(href, unsafe_allow_html=True)

        with col3:
            if st.button('Export model configuration (.json)'):
                with st.spinner("Exporting..",key="json"):
                    with open('serialized_model.json', 'w') as fout:
                        json.dump(model_to_json(m), fout)  

if page == "About":
    st.image("prophet.png")
    st.header("About")
    st.markdown("Official documentation of **[Facebook Prophet](https://facebook.github.io/prophet/)**")
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.write("")
    st.write("Author:")
    st.markdown(""" **[Giancarlo Di Donato](https://www.linkedin.com/in/giancarlodidonato/)**""")
    st.markdown("""**[Source code](https://github.com/giandata/forecast-app)**""")

    st.write("Created on 27/02/2021")
    st.write("Last updated: **01/03/2021**")
