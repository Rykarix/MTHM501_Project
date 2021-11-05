<style>
details > summary {
  font-family: "fira code";
  font-size: 14px;
  padding: 4px;
  width: 200px;
  background-color: #;
  border: none;
  box-shadow: 1px 1px 2px #bbbbbb;
  cursor: pointer;
}

details > p {
  font-family: "fira code";
  background-color: #;
  padding: 2px;
  margin: 0;
  box-shadow: 1px 1px 2px #bbbbbb;
}
</style>

## 17/18th
Task 1: Find the data I want to work with.

Found. House prices. More specifically the paid price data for housing across the UK since 1995.

This is where I encountered my first problem: the file is ~4.2Gb.

# Problem 1 - Large(ish) sized datasets:
### Modin
After doing some research I found a library that might solve my issues and concerns regarding speed & memory consumption: Modin https://modin.readthedocs.io

But the problem I had here is using it with streamlit. One of the core features of streamlit is that it runs persistent code through a browser. Therefore any library that uses complex multithreading & has its own set of 'states' to manage. Due to time constraints & other looming deadlines I decided to persue a different line of thinking. That said it's still a worthy consideration if planning something on a production level.

### File formats

My second line of thinking was to simply download smaller chunks, let's say yearly data, and work with that. After some exploration I decided to go with this.

## Solution 1 - Downloading multiple files

So first we needed to get the data. From the gov.uk website, it is easy to see that the links to files we need change only by their respective year.

http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2021.csv

http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2020.csv

So the first task was to create create code that automatically downloads the CSV for each year.

---


<details>

<summary style="font-size:14px"><b>View code:</b> Downloader

</summary><p>

---
```py
def price_paid_csv_downloader(start_year=1995, end_year=int(datetime.now().strftime("%Y"))):
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/"

    # Download from current year backwards until 1995 or error
    for year in reversed(range(start_year, (1 + int(end_year)))):
        local_filename = os.path.join("data/" + "pp-" + str(year) + ".csv")
        full_url = base_url + "pp-" + str(year) + ".csv"

        # Only download if file doesn't exist:
        if not os.path.isfile(local_filename):
            print(f"#Downloading file: {local_filename}")
            with requests.get(full_url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        else:
            print(f"#File already exists: {local_filename}")
```
---
</p></details>

---

## The Data

<details>

<summary style="font-size:14px"><b> View Code:</b> data

</summary><p>



---
```py

```
---
</p></details>

---

### 

## 24th
Solved the previous issues by downloading each year and concatinating the data I needed into a single dataframe stored as a feather file

## 
Next I needed to find a way to compile a complete list of longditudes & latitudes for use with mapbox visualisation
For this I need to

# TODO: Create function to pass Town/City string & return Long & Lat
## NOTE: Function will use geopy
### DONE: We need to restrict the rate in which we call geopy or risk hammering servers & getting IP banned
### DONE: Due to above restriction, create function that caches found places.


