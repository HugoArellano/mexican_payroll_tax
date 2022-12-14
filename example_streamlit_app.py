#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:37:45 2022

@author: hugo_arellano
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.title('Brief view of tax competition')



st.subheader('Dataset')
df = pd.read_csv('/home/hugo_arellano/Carlo/portfolio/panel_data_u20.csv')
st.write(df)


st.subheader('Exploring the basics')

if st.checkbox("Dataframe shape"):
    	st.write(df.shape)

if st.checkbox("Describe dataset"):
    	st.write(df.describe())
        
if st.checkbox('Correlation map'):
    plt.figure(figsize=(25,25))
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax, annot=True, cmap='rocket_r', 
                annot_kws={"size": 4}, fmt='.2f')
    st.write(fig)

st.subheader('States comparison')

def plot():
    
    df = pd.read_csv('/home/hugo_arellano/Carlo/portfolio/panel_data_u20.csv')
    state_list = df["ent_name"].unique().tolist()

    select_var = st.selectbox('select variable', ['isn_tasa','isn','base_gravable_formal','tasa_efec_combinada','deuda_ent',
                      'pob_ocup','impuestos_ent','ing_formal','pct_pob_formal','pct_pob_informal',
                      'ing_informal','pct_gobierno','pct_grandes',
                      'pct_medianos','pct_micronegocios','pct_otros','pct_pequeños'])
    states = st.multiselect('Select state', state_list)

    
    dfs = {state: df[df["ent_name"] == state] for state in states}

    fig = go.Figure()
    for state, df in dfs.items():
        fig = fig.add_trace(go.Scatter(x=df["anio"], y=df[select_var], name=state,
                                       ))
    fig.update_layout(title_text='States comparison of {}'.format(select_var))
    fig.update_xaxes(title_text='year')
    fig.update_yaxes(title_text='{}'.format(select_var))
    st.plotly_chart(fig)
    
plot()

st.subheader('States by year table')

selected_values = st.selectbox('select variable',['isn_tasa','isn','base_gravable_formal','tasa_efec_combinada','deuda_ent',
                      'pob_ocup','impuestos_ent','ing_formal','pct_pob_formal','pct_pob_informal',
                      'ing_informal','pct_gobierno','pct_grandes',
                      'pct_medianos','pct_micronegocios','pct_otros','pct_pequeños'], key=100)

st.write(pd.pivot_table(df,values=selected_values,index=['ent_name'],
               columns=['anio']))


st.subheader('Nominal ISN Comparison')

import folium
from streamlit_folium import folium_static
import numpy as np

json1 = 'Downloads/states_mexico.geojson'
    
m = folium.Map(location=[22.8510,-102.1255], tiles='CartoDB positron',
                    name='Light Map', zoom_start =5,
                    attr='My Data Atribution')
    
isn5y = ('/home/hugo_arellano/Carlo/portfolio/isn_each_5y.csv')
isn_data = pd.read_csv(isn5y)
    
myscale =np.linspace(isn_data['2006'].min(), isn_data['2021'].max(),4 )
    
choice = ['2006','2011','2016','2021']
choice_selected = st.selectbox('Selecciona año', choice)
folium.Choropleth(geo_data=json1,
        name='choropleth',
        data=isn_data,
        columns=['state_code', choice_selected],
        key_on='feature.properties.state_code',
        fill_color='OrRd',
        threshold_scale=myscale,
        fill_opacity=0.7,
        line_opacity=.1,
        legend_name=choice_selected).add_to(m)
folium.GeoJson('Downloads/states_mexico.geojson', name='LSOA Code',
               popup=folium.GeoJsonPopup(fields=['state_code'])).add_to(m)
folium_static(m, width=800, height=500)



st.subheader('')

    

    




