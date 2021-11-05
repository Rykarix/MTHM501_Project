import re, sys, os
from io import StringIO
from datetime import datetime  # Primarily used to reformat datetime strings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import requests

import numpy as np
import pandas as pd


pd.set_option("precision", 0)
pd.set_option("chop_threshold", 0.05)
pd.set_option("colheader_justify", "right")
pd.options.display.float_format = "{:,.2f}".format

import streamlit as st

st.set_page_config(page_title="Project Diary", layout="wide")


@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def main_df():
    try:
        df = pd.read_feather("data/combined.feather")
        return df
    except:
        combined_url = "https://github.com/Rykarix/MTHM501_Project/blob/master/data/combined.feather"
        df = pd.read_feather(request.FILES[combined_url])
        return df


@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def av_df():
    try:
        df_av = pd.read_feather("data/df_av.feather")
        return df_av
    except:
        av_url = "https://github.com/Rykarix/MTHM501_Project/blob/master/data/df_av.feather"
        df_av = pd.read_feather(request.FILES[av_url])
        return df_av


@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def main_agg(df, agg_dct):
    return df.groupby(["County", "Property Type", "Year"]).agg(agg_dct)


st.markdown(
    """


# Introduction & Objectives
Are houses more expensive now than in the past and, if so, why? Is it to do with inflation? If so, to what degree? Are there other factors involved, if so, what could they be?

The objective of this report is to explore these questions.

# Data & Initial difficulties
The data used in this report has come from the governments price paid dataset and dates back to 1995, https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads#using-or-publishing-our-price-paid-data. This is a large dataset & includes data for over 26million house sales & collective file size of 4.2Gb which, to my frustration, became one of many obstacles in creating this report.

## Collating the price paid housing data
After creating a function to download every yearly csv file from the gov website I needed to read the data. After reading this https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d I decided to save everything into a feather format. This allowed me to turn a 4.2GB file into a 565Mb file with much faster read/write times.

"""
)

df = main_df()
st.dataframe(df.head())

with st.expander("Explanation of column names & excluded data"):
    explanation_col, exclusion_col = st.columns(2)
    explanation_col.markdown(
        """
    ### Price
    Sale price stated on the deed

    ### Property Type
    D = Detatched, S = Semi-Detached, T = Terraced, F = Flats/Maisonettes, O = Other

    ### New
    true = a newly built property
    false = an established residential building

    ### Standard price paid: (Formerly "PPD Category type")
    true = Standard Price Paid entry, includes single residential property sold for value.

    false = Additional Price Paid entry including transfers under a power of sale/repossessions, buy-to-lets (where they can be identified by a Mortgage). transfers to non-private individuals and sales where the property type is classed as ‘Other’.

    ### Year

    Data taken from: https://www.gov.uk/guidance/about-the-price-paid-data#data-excluded-from-price-paid-data
    """
    )

    exclusion_col.markdown(
        """
            ### Data excluded from Price Paid Data

    Our Price Paid Data includes information on all residential property sales in England and Wales that are sold for value and are lodged with us for registration.

    Our Price Paid Data excludes:
    * sales that have not been lodged with HM Land Registry
    * sales that were not for value
    * transfers, conveyances, assignments or leases at a premium with nominal rent, which are:
        * ‘Right to buy’ sales at a discount
        * subject to an existing mortgage
        * to effect the sale of a share in a property, for example, a transfer between parties on divorce
        * by way of a gift
        * under a compulsory purchase order
        * under a court order
        * to Trustees appointed under Deed of appointment
    * Vesting Deeds Transmissions or Assents of more than one property
    """
    )

st.markdown(
    """

From here we can explore the data.

# Analysis & Results
## Price paid v Inflation
So has the average price increased over time :
"""
)

#%%
# df_av = df.groupby("Year").mean()
# df_av = df_av.reset_index()
# df_av.to_feather("data/df_av.feather")
df_av = av_df()
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_av["Year"],
        y=df_av["Price"],
        mode="lines+markers",
        name="Percent Growth (House)",
    )
)
st.plotly_chart(fig, use_container_width=True)


st.markdown(
    """

or we can can hone in on specific parameters by using a MultiIndex group aggrigates:

### Average Price of Detatched houses in Devon with Linear Regression modelling & error bars

(Warning: if there are only a few entries then the graph will look funky so this only works well for 5+ years)

"""
)


# Average Price of Detatched houses in Devon with Linear Regression modelling & error bars
# Aggrigate the data using .groupby(lst).agg(dct)
agg_dct = {
    ("Price"): [
        "count",
        "min",
        "quantile",
        "max",
        "mean",
        "std",
        "var",
        "skew",
        "sem",
    ]
}
df_agg = main_agg(df, agg_dct)

mask = df_agg.reset_index()
county_tuple = tuple(np.unique(np.array(mask["County"])))
ptype_tuple = tuple(np.unique(np.array(mask["Property Type"])))

selectbox_county = st.selectbox("Choose a County", (county_tuple), index=38)
selectbox_ptype = st.selectbox("Choose a Property type", (ptype_tuple))

st.write("Selected County: ", selectbox_county)
st.write("Selected property type: ", selectbox_ptype)
# Specify the wanted parameters
df_devon_d = df_agg.xs((str(selectbox_county), str(selectbox_ptype))).reset_index()
year_list = list(np.unique(np.array(df_devon_d["Year"])))
# Slice the results we want:
idx = pd.IndexSlice
df_devon_d = df_devon_d.loc[
    idx[:],
    idx[
        "Price",
        (
            "count",
            "min",
            "quantile",
            "max",
            "mean",
            "std",
            "var",
            "skew",
            "sem",
        ),
    ],
]
# Flatten into a single index
df_devon_d.columns = ["_".join(col) for col in df_devon_d.columns.values]
df_devon_d["Year"] = year_list
# Finally plot with linear regression:
X = df_devon_d["Year"].values.reshape(-1, 1)
model = LinearRegression()
model.fit(X, df_devon_d["Price_mean"])
x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))
df_devon_d["e"] = df_devon_d["Price_std"]
fig = px.scatter(
    df_devon_d,
    x="Year",
    y="Price_mean",
    opacity=0.65,
    error_y="e",
    title="Average price of Detatched houses in Devon with Linear Regression modelling",
)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name="Regression Fit"))
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """

## Per county comparisons

Comparison between different states & the radius of each point corresponds to the number of sales 
"""
)

multiselect_county = st.multiselect(
    "Chose which counties you wish to compare", county_tuple, default=("CARDIFF", "DEVON", "GREATER LONDON", "SURREY")
)
# Comparison of multiple counties with regression modelling
mask = df_agg.xs((slice(None), str(selectbox_ptype)))
# Slice the results we want:
idx = pd.IndexSlice
# Handle MultiIndex levels
df_county_compare = mask.loc[
    idx[mask.index.get_level_values("County"), mask.index.get_level_values("Year")],
    idx[:],
]
# Flatten
df_county_compare = df_county_compare.reset_index()
mask = df_county_compare
mask.columns = ["_".join(col) for col in df_county_compare.columns.values]
df_county_compare = mask
# Filter the dataframe based on options in multiselect_county
df_county_compare = df_county_compare[df_county_compare["County_"].isin(multiselect_county)]
county_tuple_compare = tuple(np.unique(np.array(df_county_compare["County_"])))
# Plot
fig = px.scatter(
    df_county_compare,
    x="Year_",
    y="Price_mean",
    size="Price_count",
    color="County_",
    opacity=0.65,
    title="Comparison of multiple counties with regression modelling",
    trendline="ols",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
According to various sources, https://inflation.iamkate.com/, the value of the GBP has more than doubled between 1995 & 2020.

At a glance one can see that inflation exhibits a linear growth whereas house prices exhibit a mix of exponential and linear growth



# Limitations
See https://www.gov.uk/guidance/about-the-price-paid-data#data-excluded-from-price-paid-data

## Excluded from the dataset by the source
There are exclusions in the data, as explained on the governments website. These exclusions are:

* sales that have not been lodged with HM Land Registry
* sales that were not for value
* transfers, conveyances, assignments or leases at a premium with nominal rent, which are:
    * ‘Right to buy’ sales at a discount
    * subject to an existing mortgage
    * to effect the sale of a share in a property, for example, a transfer between parties on divorce
    * by way of a gift
    * under a compulsory purchase order
    * under a court order
    * to Trustees appointed under Deed of appointment
* Vesting Deeds Transmissions or Assents of more than one property


## Parameters (columns) that are included in the mean
The purpose of this section is to provide transparency & demonstrate intillectual honesty in the report process. It may also highlight areas of improvement.

The scope of this report is to analyse the average price paid between counties and potentially towns. Therefore the following paremeters were not considered due to lack of relevance to the scope:
District, Locality, Street, SAON, PAON, Postcode, Transaction ID.

Also, due to the amount of data in this dataframe, I needed to strip as many parameters as I could get away with

### Record status
This flags whether there were additions or changes to the original price. I chose to ignore this parameter for two reasons. Lack of information & lack of time due to report deadline. The main lack of information is how each 'flag' affects the price. Given more time I could analyse the difference between these flags & infer how each flag may impact factors like standard deviation & variance. This would be useful for error analysis but beyond the scope of this report.

### Duration
This parameter highlights whether a property is a 'Freehold' or 'Lease'. This was also not considered for the same reasons as above

# Conclusion
The aim of this project was to investigate housing prices since 1995 & analyze the factors that influence the average price.

# Further research & Development
To take this report further, I'd plot the mean price onto a geospacial map of the UK, perform time analysis & attempt to create a preduction algorithm that also takes into account influencing factors such as inflation, market manipulation, supply & demand and more. Another line of enquiry would be to also analyse the cost of basic foods, toiletries & other 'essential' costs for living to analyse & compare against minimum living wage.

# References
Government PP Data: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

# Libraries used
Geopy - https://geopy.readthedocs.io/
"""
)

# %%
