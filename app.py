import streamlit as st 
import pandas as pd 
import pickle
import glob
stat = st.empty()
st.title('Air Quality Index Project')
st.text('Using Machine learning')

page = st.sidebar.selectbox('Navigation',['Prediction','Model Performance'])
if page == 'Model Performance':
    df = pd.read_csv('model_perf.csv')
    st.dataframe(df)
    for i in df.columns[1:]:
        st.header(i)
        st.bar_chart(df[[i]])
    
    
if page ==  'Prediction':
    input_meth = st.sidebar .selectbox('Select Method Of Input',['Form','File'])
    if input_meth=='Form':
        
        models_avail = [x.split('/')[-1].replace('.pkl','') for x in glob.glob('models/*.pkl')]
        model_select = st.sidebar.selectbox('Choose Model',models_avail)
        with st.form(key='aqi_form'):
            cols = st.columns(3)
            T=cols[0].number_input('Average Temperature (°C)',step=0.001)
            TM=cols[0].number_input('Maximum Temperature (°C)',step=0.001)
            Tm=cols[0].number_input('Minimum Temperature (°C)',step=0.001)
            SLP=cols[1].number_input('Atmospheric pressure at sea level (hPa)',step=0.001)
            H=cols[1].slider('Average relative humidity (%)', 0.1, 100.0,step=0.1)
            VV=cols[2].number_input('Average visibility (Km)',step=0.001)
            V=cols[2].number_input('Average wind speed (Km/h)',step=0.001)
            VM=cols[2].number_input('Maximum sustained wind speed (Km/h)',step=0.001)
            submit_button = st.form_submit_button(label='Predict')
        
            if submit_button:
                loaded_model=pickle.load(open(f'models/{model_select}.pkl', 'rb'))
            
                my_prediction = loaded_model.predict([[T,TM,Tm,SLP,H,VV,V,VM]])
                
                st.success(f'Predicted Air Quality is {round(my_prediction[0],3)} ug/m3')
        
        
        
        
    elif input_meth=='File':
        csv = st.sidebar.file_uploader('Upload a historical .csv file')
        models_avail = [x.split('/')[-1].replace('.pkl','') for x in glob.glob('models/*.pkl')]
        model_select = st.sidebar.selectbox('Choose Model',models_avail)
        
        if csv:
            df = pd.read_csv(csv)
            preview_file = st.empty()
            show = st.sidebar.checkbox('Show')
            disp =st.sidebar.radio('Visualize',['Tabular','Chart'])
            

            loaded_model=pickle.load(open(f'models/{model_select}.pkl', 'rb'))
            stat.success('Model Loaded : '+model_select)
            
            if show:
                preview_file.dataframe(df.head())
            my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
            my_prediction=my_prediction.tolist()    
            chart_data = pd.DataFrame(my_prediction,columns=['Prediction (ug/m3)'])
            if disp == 'Chart':
                st.header('Prediction')
                st.area_chart(chart_data)
            elif disp == 'Tabular':
                st.header('Prediction')
                st.dataframe(chart_data)
            
        
        
    
