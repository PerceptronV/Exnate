# Dataloading

This repo contains the source code to load financial data for the Exnate project. These data include:
* various pairs exchange rates 

* various important index stocks in the UK and US

* international financial statistics (e.g. interest rates, GDP)
  

## Sources

* Exchange rates and index stocks data are scraped from [investing.com](https://www.investing.com/), with the use of [investpy](https://pypi.org/project/investpy/)

* international financial statistics are taken from 2 IMF (International Monetary Fund) databases
    
    1. The [IMF World Economic Outlook Database 2020](https://www.imf.org/en/Publications/WEO/weo-database/2020/October) via a downloaded version in this repo ([weo_data_oct_2020.csv](weo_data_oct_2020.csv))
    
    2. The [IMF IFS (International Financial Statistics) dataset](https://data.imf.org/?sk=4C514D48-B6BA-49ED-8AB9-52B0C1A0179B) via IMF's [JSON API](http://datahelp.imf.org/knowledgebase/articles/667681-using-json-restful-web-service)
    

## Usage

### Prerequisites

* pandas

* investpy

* tqdm

### Using the dataloader

1. Download [dataloader.py](dataloader.py)
   
2. From [dataloader.py](dataloader.py) import `get_features` (returns a Pandas Dataframe and a dictionary of features) or `get_save` (saves the downloaded data and features dictionary)

3. In the `get_features` or `get_save` functions, specify start date and end date with Python datetime objects; specify additional argument `beta=True` if you'd like to download IMF IFS dataset too
    
    * Fully downloaded features are already available: [full_data.csv](full_data.csv) and [full_data_beta.csv](full_data_beta.csv);
    
    * together with their respective feature dictionaries, mapping the _i_ th column to its feature name: [feature_names.json](feature_names.json), [feature_names_beta.json](feature_names_beta.json)
    
    * _Due to a bug, the keys in the feature dictionaries are strings of the _i_ th feature, not numbers_
    
## Data Structure

The Pandas Dataframe returned by `get_features` (or loaded from the csv files generated from `get_save`) is of the following structure:

| | [Name of _feature 0_] | [Name of _feature 1_] | _..._ | [Name of _feature i_] |
| :---: | :---: | :---: | :---: | :---: |
| __index__ | | | | |
| _yyyy-mm-dd_ | [Value of _feature 0_] | [Value of _feature 1_] | _..._ | [Value of _feature i_] |
| _yyyy-mm-dd_ | [Value of _feature 0_] | [Value of _feature 1_] | _..._ | [Value of _feature i_] |
| _..._ | | | _..._ | |
| _yyyy-mm-dd_ | [Value of _feature 0_] | [Value of _feature 1_] | _..._ | [Value of _feature i_] |

The dictionary of features, returned as the second value from `get_features` (or loaded from the json files generated from `get_save`) is of the following structure:

```json5
{
  "0": "Name of feature 0",
  "1": "Name of feature 1",
  ...
  "i": "Name of feature i"
}
```
