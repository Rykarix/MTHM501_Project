# vscode-fold=1

#%%
import re, sys, os, logging
from io import StringIO
from datetime import datetime  # Primarily used to reformat datetime strings
from IPython.display import clear_output, display, HTML
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def myprint(fstring):
    clear_output(wait=True)
    sys.stdout.write("\r" + fstring)
    sys.stdout.flush()


logging.basicConfig(filename="tools.log", encoding="utf-8", level=logging.DEBUG)

import pandas as pd

pd.set_option("precision", 0)
pd.set_option("mode.sim_interactive", True)
pd.set_option("chop_threshold", 0.05)
pd.set_option("colheader_justify", "right")
# pd.set_option("float_format", "{:,.2f}".format)
pd.options.display.float_format = "{:,.2f}".format

import numpy as np

# from numba import jit
from geopy.geocoders import (
    Nominatim,
)  # Library for returning Longtidtude & Latitude of geolocations - see https://geopy.readthedocs.io/
from geopy.extra.rate_limiter import (
    RateLimiter,
)  # Used to limit the rate of requests so we don't get IP Banned
from geopy.geocoders import get_geocoder_for_service
import geopy
import pickle
import sqlite3
import pprint
import pydeck as pdk

from tqdm import tqdm


class Cache(object):
    def __init__(self, fn="cache.db"):
        self.conn = conn = sqlite3.connect(fn)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS " "Geo ( " "town STRING PRIMARY KEY, " "location BLOB " ")")
        conn.commit()

    def town_cached(self, town):
        cur = self.conn.cursor()
        cur.execute("SELECT location FROM Geo WHERE town=?", (town,))
        res = cur.fetchone()
        if res is None:
            return False
        return pickle.loads(res[0])

    def save_to_cache(self, town, location):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO Geo(town, location) VALUES(?, ?)",
            (town, sqlite3.Binary(pickle.dumps(location, -1))),
        )
        self.conn.commit()


class Geo:
    """Geopy search object

    Args:
        GeoCoder (string): The geocoder to use, ie: Baidu, Bing, GoogleV3, TomTom etc
        Config (dict): the __init__ parameters needed for the above configs
    Returns:
        Tuple: (Longitude, Latitude)
    """

    def __init__(self, geocoder, config, p=False):
        self.geocoder = geocoder
        self.config = config
        self.p = p
        if geocoder != "Nominatim":
            raise Exception("Sorry but Nominatim is currently the only supported geocoder for this function")

    def name_to_gps(self, data_frame, column_name, geo_app, saveAsFeather=False):
        """get the gps coords of a place

        Args:
            data_frame (class): Pandas dataframe
            column_name (string): Name of column with towns or cities
            saveAsFeather (bool, optional): Should result be saved as a file. Defaults to False.

        Returns:
            class: Pandas dataframe
        """
        data_frame["location"] = data_frame[column_name].apply(geo_app)
        data_frame["point"] = data_frame["location"].apply(lambda loc: tuple(loc.point) if loc else None)
        return data_frame

    def gps(self, query):

        # use set geocoder
        cls = get_geocoder_for_service(self.geocoder)

        # apply configs
        geolocator = cls(**self.config)

        # ensure everything is lowercase
        town = query.lower()

        # database checking function
        in_db = cache.town_cached(town)

        # if in db, return gps, else save to db and return gps
        if in_db:
            # print("was cached: {}\n{}, {}".format(in_db, in_db.longitude, in_db.latitude))
            return (float(in_db.longitude), float(in_db.latitude))

        else:
            # print("was not cached, looking up and caching now")

            # limit the rate we make requests
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

            # limit the results to places in the uk
            location = geocode(town, country_codes="gb")

            # print("found as: {}\n{}, {}".format(location, location.longitude, location.latitude))

            # save to cache
            cache.save_to_cache(town, location)
            # print("... and now cached.")
            return (float(location.longitude), float(location.latitude))

    def lat(self, query):

        # use set geocoder
        cls = get_geocoder_for_service(self.geocoder)

        # apply configs
        geolocator = cls(**self.config)

        # ensure everything is lowercase
        town = query.lower()

        # database checking function
        in_db = cache.town_cached(town)

        # if in db, return gps, else save to db and return gps
        if in_db:
            if self.p:
                myprint("CACHED: Latitude for {} already cached: {}".format(in_db, in_db.latitude))
            return in_db.latitude

        else:
            # print("was not cached, looking up and caching now")

            # limit the rate we make requests
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

            # limit the results to places in the uk
            location = geocode(town, country_codes="gb")
            if self.p:
                myprint("NOT CACHED: Latitude for {} found: {}".format(location, location.latitude))
            # save to cache
            try:
                cache.save_to_cache(town, location)
            except AttributeError:
                pass
            # print("... and now cached.")

            return location.latitude

    def lon(self, query):

        # use set geocoder
        cls = get_geocoder_for_service(self.geocoder)

        # apply configs
        geolocator = cls(**self.config)

        # ensure everything is lowercase
        town = query.lower()

        # database checking function
        in_db = cache.town_cached(town)

        # if in db, return gps, else save to db and return gps
        if in_db:
            if self.p:
                myprint("CACHED: Longitude for {} already cached: {}".format(in_db, in_db.longitude))
            return in_db.longitude

        else:
            # print("was not cached, looking up and caching now")

            # limit the rate we make requests
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

            # limit the results to places in the uk
            location = geocode(town, country_codes="gb")
            if self.p:
                myprint("NOT CACHED: Longitude for {} found: {}".format(location, location.longitude))

            # save to cache
            try:
                cache.save_to_cache(town, location)
            except AttributeError:
                pass
            # print("... and now cached.")
            return location.longitude

    def bb(self, query):

        # use set geocoder
        cls = get_geocoder_for_service(self.geocoder)

        # apply configs
        geolocator = cls(**self.config)

        # ensure everything is lowercase
        town = query.lower()

        # database checking function
        in_db = cache.town_cached(town)

        # if in db, return gps, else save to db and return gps
        if in_db:
            if self.p:
                myprint("CACHED: Bounding Box for {} already cached: {}".format(in_db, in_db.raw["boundingbox"]))
            return in_db.raw["boundingbox"]

        else:
            # print("was not cached, looking up and caching now")

            # limit the rate we make requests
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

            # limit the results to places in the uk
            location = geocode(town, country_codes="gb")
            if self.p:
                myprint("NOT CACHED: boundingbox for {} found: {}".format(location, location.raw["boundingbox"]))

            # save to cache
            try:
                cache.save_to_cache(town, location)
            except AttributeError:
                pass
            # print("... and now cached.")
            return location.raw["boundingbox"]


def price_paid_csv_downloader(start_year=1995, end_year=int(datetime.now().strftime("%Y")), folder="data/"):
    # http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2021.csv

    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/"

    # Download from current year backwards until 1995 or error
    for year in reversed(range(start_year, (1 + int(end_year)))):
        local_filename = os.path.join(folder, "pp-" + str(year) + ".csv")
        full_url = base_url + "pp-" + str(year) + ".csv"

        # Only download if file doesn't exist:
        if not os.path.isfile(local_filename):
            myprint(f"#Downloading file: {local_filename}")
            with requests.get(full_url, stream=True) as r:

                # Raise an HTTP error if one occurs
                r.raise_for_status()
                with open(local_filename, "wb") as f:

                    # Download in chunks
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        else:
            myprint(f"#File already exists: {local_filename}")


def prepare_df(csv_folder="data/range/"):
    """Import data from multiple csv files

    Args:
        csv_folder (str, optional): path to flder. Defaults to "data/range/Current".
        convert_csv_to_df (bool, optional): If set to True, save all csv files as feather after cleaning data with csv_to_df. Defaults to False.
    """

    def csv_to_df(file_path):
        """Extracts & formats each column of a csv file into a data frame

        Args:
            file_path (string): Example: "data/range/pp-1995.csv"

        Returns:
            class: Pandas DataFrame
        """
        data_frame = pd.read_csv(
            str(file_path),
            usecols=[1, 2, 4, 5, 11, 13, 14],
            names=[
                # "Transaction unique identifier",  #     0  No
                "Price",  #                             1
                "Date of Transfer",  #                  2
                # "Postcode",  #                          3
                "Property Type",  #                     4
                "New",  #                           5
                # "Duration",  #                          6
                # "PAON",  #                              7  No
                # "SAON",  #                              8  No
                # "Street",  #                            9
                # "Locality",  #                          10 No
                "Town",  #                         11
                # "District",  #                          12
                "County",  #                            13
                "SPP",  #                 14
                # "Record Status - monthly file only",  # 15
            ],
            error_bad_lines=False,  # Needed for a pp-2010.csv, there seems to be some form of corruption on that line
            low_memory=True,
            memory_map=True,
            engine="python",
        )
        # Add year column as an unsigned 16 bit integer
        data_frame["Year"] = (
            pd.to_datetime(
                data_frame["Date of Transfer"],  #
                format="%Y-%m-%d %H:%M",  # Format that string date is in CSV
                dayfirst=True,  # Day is first in format so specifying this
                errors="coerce",  # Any invalid date formats will return NaT
            )
            .dt.strftime("%Y")
            .astype("uint16")
        )

        # Remove Datetime column, we only need the year
        data_frame.drop("Date of Transfer", axis=1, inplace=True)

        # Convert SPP into boolean
        data_frame["SPP"] = np.where(data_frame["SPP"] == "A", True, False)

        # Convert New to boolean
        data_frame["New"] = np.where(data_frame["New"] == "Y", True, False)

        # Convert Price to float32
        data_frame["Price"] = data_frame["Price"].astype("uint64")

        # Convert Property Type to string
        data_frame["Property Type"] = data_frame["Property Type"].astype("string")

        # Convert County to string
        data_frame["County"] = data_frame["County"].astype("string")

        # Convert Town to string
        data_frame["Town"] = data_frame["Town"].astype("string")
        myprint(f"{file_path} successfully imported to data frame")
        return data_frame

    def csv_list(csv_folder):
        # return a sorted list of files
        file_list = []

        list_of_files = sorted(
            filter(lambda x: os.path.isfile(os.path.join(csv_folder, x)), os.listdir(csv_folder)), reverse=True
        )

        for i in list_of_files:
            # only append csv files
            if i.lower().endswith(".csv"):
                file_list.append(os.path.join(i))

        return file_list

    def feather_list(feather_folder):
        # return a sorted list of files
        file_list = []

        list_of_files = sorted(
            filter(lambda x: os.path.isfile(os.path.join(feather_folder, x)), os.listdir(feather_folder)), reverse=True
        )

        for i in list_of_files:
            # only append csv files
            if i.lower().endswith(".feather"):
                file_list.append(os.path.join(i))

        return file_list

    if not os.path.isfile("data/combined.feather"):
        myprint(f"combined.feather does not exist, creating it now...")

        feather_folder = os.path.join(csv_folder, "feathers")

        for csv_file in csv_list(csv_folder):
            csv_file_path = os.path.join(csv_folder, csv_file)

            feather_file = str(csv_file + ".feather")
            feather_file_path = os.path.join(feather_folder, feather_file)
            myprint(f"Importing{feather_file_path}")

            # Check if feather file doesn't exist
            if not os.path.isfile(feather_file_path):
                myprint(f"{feather_file_path} doesn't exist, creating now...")

                # Convert data frame to feather
                df_to_feather(
                    # Convert CSV File to cleaned data frame
                    csv_to_df(csv_file_path),
                    folder_path=feather_folder,
                    file_name=feather_file,
                )
                myprint(f"{feather_file_path} saved")
            else:
                myprint(f"#File already exists: {feather_file_path}")
        # Concatinate all the created feathers into a single dataframe
        data_frame = pd.concat(map(feather_to_df, feather_list(feather_folder)), ignore_index=True)

        # Save concatinated dataframe to feather file
        myprint(f"Saving combined dataframe to: data/combined.feather")
        df_to_feather(data_frame)
        myprint(f"data/combined.feather saved")
        # return the dataframe
        return data_frame
    else:
        myprint(f"data/combined.feather already exists. Importing now:")
        data_frame = feather_to_df(feather_file="combined.feather", feather_folder="data/")
        myprint(f"data/combined.feather Imported")
        return data_frame


def feather_to_df(feather_file, feather_folder="data/range/feathers/"):
    feather_path = os.path.join(feather_folder, feather_file)
    data_frame = pd.read_feather(
        str(feather_path),
    )
    return data_frame


def df_to_feather(data_frame, folder_path="data/", file_name="combined.feather"):
    # https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
    # feather seems to be the most well rounded in terms of performance & memory consumption
    data_frame.to_feather(os.path.join(folder_path, file_name), compression="lz4")
    myprint(f"{folder_path}{file_name} saved")


def reformat_datetime(data_frame, column, dtformat="%Y-%m-%d", update_file=False):
    tqdm.pandas()
    mask = tqdm(pd.to_datetime(data_frame.loc[:, "Date of Transfer"]).dt.strftime(dtformat))
    dfc = data_frame.copy()
    # Pandas does not like it when there's obfuscation between one dataframe & a copy of it.
    # See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    dfc[column] = mask
    if update_file:
        df_to_feather(dfc)
        return dfc
    else:
        return dfc


def add_gps_town(data_frame, update_file=False):

    searcher = Geo("Nominatim", {"user_agent": "get_gps"})
    tqdm.pandas()
    mask_lat = data_frame["Town"].progress_apply(searcher.lat)
    mask_lon = data_frame["Town"].progress_apply(searcher.lon)
    mask_bb = data_frame["Town"].progress_apply(searcher.bb)
    dfc = data_frame
    dfc["lat"] = mask_lat
    dfc["lon"] = mask_lon
    dfc["bb"] = mask_bb

    if update_file:
        df_to_feather(dfc)
        return dfc
    else:
        return dfc


def add_population(data_frame, update_file=False):
    pass


def datastore_to_df(path="processed_data.h5"):
    try:
        data_store = pd.HDFStore(path)
        df = data_store["df"]
        data_store.close()
        return df
    except:
        pass


def price_agg(data_frame, list_of_columns, dict_of_agg_args):
    df_groupby = data_frame.groupby(list_of_columns)
    return df_groupby.agg(dict_of_agg_args)


def my_sort(data_frame):
    df = data_frame.groupby(
        [
            "County",
            "Town",
            "Property Type",
            "Old/New",
            "Standard Price Paid",
            "Year of Transfer",
        ]
    )
    # .agg({"Price": ["min", "max", "mean", "count", "std", "var", "skew", "sem"]})
    return df


#%%
# Download all csv's into a folder
# price_paid_csv_downloader(start_year=1995, end_year=2021, folder="data/range/")

#%%
# Extract wanted data from folder of csvs, save each year as feather, concatinate each feather file into a single dataframe. This turns 4.2Gb of data into 565Mb
# df = prepare_df()


#%%
# Group by & aggrigate data - This can take a while since we're dealing with 26 million rows. May not work on low spec machines?
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
# lst = ["County", "Town", "Year", "Property Type", "New", "SPP"]
# df = df.groupby(lst).agg(dct)
# df

#%%
# Load data from file and examine the end result:
df = pd.read_feather("data/combined.feather")
df
# We now have 314,583 rows and 15 columns containing all the data we need for visual analysis

#%% === PP v INFLATION
# Average house price since 1995 v Inflation
# Get the mean data
df_av = df.groupby("Year").mean()
df_av = df_av.reset_index()
# Get the inflation data
df_inflation = pd.read_csv("data/inflation.csv")
df_inflation = df_inflation.sort_values("Year", ascending=True)
# Merge the data
df_compare = df_av
df_compare["Mult"] = df_inflation["Mult"]
df_compare = df_compare.dropna()
# Convert Price to a percentage since 1995
df_compare["Price Percentage"] = df_compare["Price"] / 67932.26
df_compare
# Plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df_compare["Year"], y=df_compare["Price Percentage"], mode="lines+markers", name="Percent Growth (House)")
)
fig.add_trace(
    go.Scatter(x=df_compare["Year"], y=df_compare["Mult"], mode="lines+markers", name="Percent Growth (Inflation)")
)
fig.show()

#%% === PP + DEVON + DETATCHED + LINEAR REGRESSION
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
df_agg = df.groupby(["County", "Property Type", "Year"]).agg(agg_dct)
# Specify the wanted parameters
df_devon_d = df_agg.xs(("DEVON", "D")).reset_index()
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
df_devon_d["Year"] = [year for year in range(1995, 2022)]
df_devon_d
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
fig.show()

#%%
df_devon_d

#%%
# Comparison of multiple counties with regression modelling
mask = df_agg.xs((slice(None), "D"))
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
df_county_compare
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
fig.show()


#%%
test = X.copy()
test.columns = ["_".join(col) for col in test.columns.values]
test

#%%
agg = test.groupby(["County", "Year"]).mean("Price")
agg
# unique df - for exploring geopy stuff
# unique = df.groupby(["County", "Town"]).size().reset_index(name="Freq")
# unique = unique.drop("Freq", axis=1)
# unique

#%%
# Load the combined_agg data from feather
df = feather_to_df(feather_file="combined_agg.feather", feather_folder="data/")
df

# %%
# Add Town coords to unique data frame

test = df[["County", "Town", "Year", "Property Type", "count", "mean"]]
test = test[test["count"] > 2]
test = test[test["Property Type"] == "D"]
test

fig = px.box(
    test,
    x="Year",
    y="mean",
    # size="count",
    color="County",
    hover_name="count",
    log_x=False,
    # size_max=500,
)
fig.show()

#%%
# read in population data
