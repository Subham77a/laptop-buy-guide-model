import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # Convert categorical Yes/No to numeric
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Ensure correct types based on df.info()
    ram = int(ram)
    weight = float(weight)
    hdd = int(hdd)
    ssd = int(ssd)
    ppi = float(ppi)

    # Build query with correct order and dtypes
    query = [[
        company,        # 0 - object
        type,           # 1 - object
        ram,            # 2 - int32
        weight,         # 3 - float32
        touchscreen,    # 4 - int64
        ips,            # 5 - int64
        ppi,            # 6 - float64
        cpu,            # 7 - object
        hdd,            # 8 - int64
        ssd,            # 9 - int64
        gpu,            # 10 - object
        os              # 11 - object
    ]]

    # Convert to numpy with dtype=object to preserve mixed types
    query = np.array(query, dtype=object)

    # Predict
    predicted_log_price = pipe.predict(query)[0]
    predicted_price = int(np.exp(predicted_log_price))

    st.title(f"The predicted price of this configuration is ₹{predicted_price}")
