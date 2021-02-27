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

import plotly as plt
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.figure_factory as ff
import base64
import itertools

st.set_page_config(page_title ="Forecast App")


tabs = ["Forecast","About"]
page = st.sidebar.radio("Tabs",tabs)




def load_csv():
    df_input=pd.read_csv(input, sep = ';',encoding='utf-8')
    df_input['ds'] = pd.to_datetime(df_input['ds'],errors='coerce')
    df_input['ds'] = df_input['ds'].dt.strftime('%m/%d/%Y')
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input

st.title('Forecast DIY 🧙🏻')
st.write('This app enables you to generate time series forecast withouth any dependencies.')
st.markdown("""The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")
df =  pd.DataFrame()   

st.subheader('1. Data loading')
st.write("The dataset shall contains 2 columns: a column with dates named **ds** and a column with the historical serie named **y**.")

input = st.file_uploader('Upload time series.')

if input:
    
    df = load_csv()
    st.info("Datos cargados correctamente")


    if st.checkbox('Show data',key='show'):
        col1, col2 = st.beta_columns(2)

        with col1:
            st.write(df)
        with col2:
            st.line_chart(df['y'],use_container_width =True,height = 300)
        
            
st.subheader("2. Parameters configuration")

with st.beta_container():
    st.write('In this section you can modify the algorithm settings.')
        
    with st.beta_expander("Horizon"):
        periods_input = st.number_input('Select how many future periods to forecast.',
        min_value = 1, max_value = 365,value=90)

    with st.beta_expander("Seasonality"):
        st.write("The seasonality depends on the specific case, therefore specific domain knoledge is required.")
        seasonality = st.radio(label='Seasonality',options=['additive','multiplicative'])

    with st.beta_expander("Trend components"):
        st.write("Add or remove components")
        daily = st.checkbox("Daily")
        weekly= st.checkbox("Weekly")
        # añadir monthly 1-31 dias
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

    with st.beta_expander('Holydays'):
        if st.checkbox('Add country holidays') is False:
            holidays == True

    with st.beta_expander('Hyperparameters'):
        st.write('In this section it\'s possible to tune the scaling coefficients.')
            
        changepoint_scale_values= [0.01, 0.1, 0.5]
        seasonality_scale_values= [0.1, 1.0, 10.0]
        changepoint_scale= st.select_slider(label= 'Changepoint prior scale',options=changepoint_scale_values)
        seasonality_scale= st.select_slider(label= 'Seasonality prior scale',options=seasonality_scale_values)


with st.beta_container():
    st.subheader("3. Crear predicción")
    st.write(' En esta sección se hace el fit del modelo a los datos y se genera la prediccion para el futuro.')
    if st.checkbox("Inicializar modelo (Fit)"):
        if len(growth_settings)==2:
            m = Prophet(seasonality_mode=seasonality,
                        daily_seasonality=daily,
                        weekly_seasonality=weekly,
                        yearly_seasonality=yearly,
                        growth=growth,
                        changepoint_prior_scale=changepoint_scale,
                        seasonality_prior_scale= seasonality_scale)
            if holidays:
                m.add_country_holidays(country_name='ES')
                
            if monthly:
                m.add_seasonality(name='monthly', period=30.5, fourier_order=4)
                
            m.fit(df)
            future = m.make_future_dataframe(periods=periods_input,freq='D')
            future['cap']=cap
            future['floor']=floor
            st.write("El modelo generará predicciones hasta el ", future['ds'].max())
                
        else:
            st.warning('Configuración invalida')

        st.success('Model fitted succesfully')
            
        if st.checkbox("Generar forecast (Predict)"):
            forecast = m.predict(future)
            st.success('Predición generada')
            st.dataframe(forecast)
            fig1 = m.plot(forecast)
            st.write(fig1)

            if growth == 'linear':
                fig2 = m.plot(forecast)
                a = add_changepoints_to_plot(fig2.gca(), m, forecast)
                st.write(fig2)

        if st.button('Mostrar componentes'):
            fig3 = m.plot_components(forecast)
            st.write(fig3)


    st.subheader('4. Model validation')
    st.write("En esta sección es posible hacer cross-validation del fit.")
    with st.beta_expander("Explanation"):
        st.markdown("""The Prophet library makes it possible to divide our historical data into training data and testing data for cross validation. The main concepts for cross validation with Prophet are:""")
        st.write("Training data (initial): The amount of data set aside for training. The parameter is in the API called initial.")
        st.write("Horizon: The data set aside for validation.")
        st.write("Cutoff (period): a forecast is made for every observed point between cutoff and cutoff + horizon.""")

        
        
    initial = st.number_input(value= 330,label="initial",min_value=120,max_value=2000)
    initial = str(initial) + " days"

    period = st.number_input(value= 180,label="period",min_value=1,max_value=365)
    period = str(period) + " days"

    horizon = st.number_input(value= 120, label="horizon",min_value=30,max_value=366)
    horizon = str(horizon) + " days"

    st.write(f"Here we do cross-validation to assess prediction performance on a horizon of **{horizon}** days, starting with **{initial}** days of training data in the first cutoff and then making predictions every **{period}** days.")
        
    with st.beta_expander("Definición metricas"):
                st.write("Mse: mean absolute error")
                st.write("Rmse: mean squared error")
                st.write("Mae: Mean average error")
                st.write("Mape: Mean average percentage error")
                st.write("Mdape: Median average percentage error")
        
    if st.checkbox('Calcular metricas'):
        df_cv = cross_validation(m, initial=initial,
                                        period=period, 
                                        horizon = horizon,
                                        parallel="processes")
        #In Python, the string for initial, period, and horizon should be in the format used by Pandas Timedelta, which accepts units of days or shorter.
        # custom cutoffs = pd.to_datetime(['2019-11-31', '2019-06-31', '2021-01-31'])
            
        df_p = performance_metrics(df_cv)
        st.dataframe(df_p)
            
        metrics = ['mse','rmse','mae','mape','mdape','coverage']
            
        selected_metric = st.radio(label='Metrica por visualizar',options=metrics)
        st.write(selected_metric)
        fig4 = plot_cross_validation_metric(df_cv, metric=selected_metric)
        st.write(fig4)

    st.subheader('5. Hyperparameter Tuning')
    st.write("En esta sección es posible encontrar la combinación mejor de hyperparamentros.")

    param_grid = {  
                        'changepoint_prior_scale': [0.01, 0.1, 0.5],
                        'seasonality_prior_scale': [0.1, 1.0, 10.0],
                    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    if st.button("Optimize hyperparameters"):

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

    st.subheader('6. Exportar datos')
        
    col1, col2, col3 = st.beta_columns(3)

    with col1:
        if st.button('Exportar forecast'):
            
            export_forecast = pd.DataFrame(forecast[['ds','yhat_lower','yhat','yhat_upper']]).to_csv()
            b64 = base64.b64encode(export_forecast.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click derecho > guardar como **forecast.csv**)'
            st.markdown(href, unsafe_allow_html=True)
        
    with col2:
        if st.button("Export metricas modelo"):
            b64 = base64.b64encode(df_p.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click derecho > guardar como **metrics.csv**)'
            st.markdown(href, unsafe_allow_html=True)

    with col3:
        if st.button('Exportar modelo'):
            with open('serialized_model.json', 'w') as fout:
                json.dump(model_to_json(m), fout)  

    