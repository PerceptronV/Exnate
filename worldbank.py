import requests


def query(countries: list, indicator: str, date: str):
    url = 'http://api.worldbank.org/v2/country/{}/indicator/{}?'.format(
        ';'.join(countries),
    )
