from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder


import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

num_cols = ['PH',	'TEMPERATURE', 'HUMIDITY', 'RAINFALL', 'Area(Hectare)']

st.subheader('Millet Prediction System')


@st.cache_data
def load_dataset():
    df = pd.read_csv("final_data.csv")
    df.dropna(inplace=True)
    df['TEMPERATURE'] = pd.to_numeric(df['TEMPERATURE'])
    return df


st.write("Data after cleaning")
df = load_dataset()
df

st.subheader('Correlation Heatmap')


@st.cache_data
def corr_heatmap(df):
    corr = df.corr()
    f, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, mask=mask, cmap=cmap)
    st.write(f)


corr_heatmap(df)


st.subheader('Scatter Plot')


@st.cache_data
def scatter_plot(df):
    fig = sns.pairplot(df, vars=["Yield(Tonnes/Hectare)",
                                 "PH", "TEMPERATURE", "RAINFALL"])
    st.pyplot(fig)


scatter_plot(df)


st.subheader('Box Plot')


@st.cache_data
def box_plot(df):
    grouped_data = df.groupby("YEAR")
    df_g = pd.DataFrame()
    df_g["YEAR"] = grouped_data["YEAR"].first()
    df_g["Yield(Tonnes/Hectare)"] = grouped_data["Yield(Tonnes/Hectare)"].mean()

    st.line_chart(
        data=df_g,
        x="YEAR",
        y="Yield(Tonnes/Hectare)")


box_plot(df)


st.subheader('Line Plot')


@st.cache_data
def line_plot(df):
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    sns.lineplot(x="TEMPERATURE", y="Yield(Tonnes/Hectare)",
                 data=df, ax=axs[0])
    sns.lineplot(x="HUMIDITY", y="Yield(Tonnes/Hectare)", data=df, ax=axs[1])
    sns.lineplot(x="RAINFALL", y="Yield(Tonnes/Hectare)", data=df, ax=axs[2])
    axs[0].set(title="Yield vs Temperature", xlabel="Temperature",
               ylabel="Yield(Tonnes/Hectare)")
    axs[1].set(title="Yield vs Humidity", xlabel="Humidity",
               ylabel="Yield(Tonnes/Hectare)")
    axs[2].set(title="Yield vs Rainfall", xlabel="Rainfall",
               ylabel="Yield(Tonnes/Hectare)")
    st.pyplot(fig)


line_plot(df)


@st.cache_data
def make_model(df):
    df.rename(
        columns={'Production (Tonnes)': 'Production(Tonnes)'}, inplace=True)
    # feature selection Drop unnecessary data
    df = df.drop(['DISTRICT', 'YEAR', 'Yield(Tonnes/Hectare)'], axis=1)
    df = df.dropna()

    # Create scaler object
    scaler = MinMaxScaler()

    # Scale numerical columns
    df[num_cols] = scaler.fit_transform(df[num_cols])


# One-hot encode categorical variable 'soil'
    ohe = OneHotEncoder()
    feats = ohe.fit_transform(df[['SOIL', 'MILLETS']]).toarray()
    cols = ohe.get_feature_names_out(['SOIL', 'MILLETS'])
    ohe_df = pd.DataFrame(feats, columns=cols)

    df = df.reset_index(drop=True)
    ohe_df.reset_index(drop=True, inplace=True)

    df = pd.concat([df, ohe_df], axis=1)
    df.drop(['SOIL', 'MILLETS'], axis=1, inplace=True)

    y = df["Production(Tonnes)"]
    X = df.drop('Production(Tonnes)', axis=1)

    # Split df into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model_xg = XGBRegressor(n_estimators=1000, max_depth=7,
                            eta=0.1, subsample=0.7, colsample_bytree=0.8)
    model_xg.fit(X_train, y_train)

    print(X_test.columns)

    # Make predictions on test df
    y_pred = model_xg.predict(X_test)

    # Evaluate model performance
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('RMSE:', rmse)

    accu = model_xg.score(X_test, y_test)
    print('Accuracy:', accu)

    return model_xg, scaler, ohe


model, scaler, encoder = make_model(df)

ph = st.number_input('PH', min_value=0.0, max_value=14.0, value=0.0)
temp = st.number_input('Temperature', min_value=0.0,
                       max_value=100.0, value=0.0)
hum = st.number_input('Humidity', min_value=0.0, max_value=100.0, value=0.0)
rain = st.number_input('Rainfall', min_value=0.0, max_value=5000.0, value=0.0)
area = st.number_input('Area(Hectare)', min_value=0.0,
                       max_value=5000.0, value=0.0)
soil = st.selectbox(
    'Soil', ['red yellow', 'red loamy', 'mixed', 'loamy', 'laterite', 'black'])
millet = st.selectbox('Millet', ['Bajra', 'Barley', 'Jowar', 'Maize', 'Ragi'])

if st.button('Predict'):
    data = [[ph, temp, hum, rain, area, soil, millet]]
    data = pd.DataFrame(data, columns=['PH', 'TEMPERATURE', 'HUMIDITY',
                                       'RAINFALL', 'Area(Hectare)', 'SOIL', 'MILLETS'])
    data[num_cols] = scaler.transform(data[num_cols])
    feats = encoder.transform(data[['SOIL', 'MILLETS']]).toarray()
    cols = encoder.get_feature_names_out(['SOIL', 'MILLETS'])
    ohe_df = pd.DataFrame(feats, columns=cols)

    data = data.reset_index(drop=True)
    ohe_df.reset_index(drop=True, inplace=True)

    data = pd.concat([data, ohe_df], axis=1)
    data.drop(['SOIL', 'MILLETS'], axis=1, inplace=True)

    prediction = model.predict(data)
    st.write('The predicted production is', prediction[0]/area, 'tonnes')
