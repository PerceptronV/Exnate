import requests
import os
import locale
import pandas as pd
from datetime import datetime as dt
from bs4 import BeautifulSoup
from tqdm import tqdm


def pad(a):
    if a < 10:
        return '0'+str(a)
    else:
        return str(a)


query = ['January odd-data', 'February even-data', 'March odd-data', 'April even-data', 'May odd-data',
         'June even-data', 'July odd-data', 'August even-data', 'September odd-data', 'October even-data',
         'November odd-data', 'December even-data']
url_base = 'https://www.poundsterlinglive.com/bank-of-england-spot/historical-spot-exchange-rates/gbp/GBP-to-HKD-'
cols = ['Year', 'Month', 'Day', 'GBP->HKD']
all_data = pd.DataFrame(columns=cols)

if os.path.exists('exchange_rate.txt'):
    os.remove('exchange_rate.txt')
if os.path.exists('exchange_rate.csv'):
    os.remove('exchange_rate.csv')
txt_file = open('exchange_rate.txt', 'w')
dat_file = open('exchange_rate.csv', 'wb')

print(locale.getlocale())

txt_file.write(' '.join(cols) + '\n')

with tqdm(range(1977, 2020)) as t:
    for year in t:
        t.set_description('Year {}'.format(year))
        page = requests.get(url_base + str(year))
        soup = BeautifulSoup(page.content, 'html.parser')

        for monthly in query:
            month = soup.find_all(class_=monthly)
            month.reverse()

            for i in range(len(month)):
                bundle = month[i].get_text().strip().split('\n')
                date = dt.strptime(bundle[0].strip(), "%a, %d %b %Y")
                rate = float(bundle[1].strip().split(' ')[3])

                all_data = all_data.append(pd.DataFrame(
                    [[date.year, date.month, date.day, rate]],
                    index=["{}-{}-{}".format(date.year, pad(date.month), pad(date.day))],
                    columns=cols
                ))
                txt_file.write("{} {} {} {}\n".format(
                    date.year, date.month, date.day, rate
                ))

all_data.to_csv(dat_file, mode='wb')

dat_file.close()
txt_file.close()

print('Data Pulling Success')
