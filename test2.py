#%%
import pandas as pd
import folium
import os
import json

state_geo = "http://geoportal1-ons.opendata.arcgis.com/datasets/687f346f5023410ba86615655ff33ca9_1.geojson"
agg = pd.read_feather("data/agg_test.feather")
UK_linear = folium.Map([51.50853, -0.12574], zoom_start=7)

#%%
folium.Choropleth(
    geo_data=state_geo,
    name="Test",
    data=agg,
    columns=["County", "Price"],
    key_on="feature.properties.ctyua16nm",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Price",
    highlight=True,
).add_to(UK_linear)

UK_linear

# %%
