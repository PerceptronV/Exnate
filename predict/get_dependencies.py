import requests

r = requests.get('https://raw.githubusercontent.com/PerceptronV/Exnate/master/data/dataloader.py')
open('dataloader.py', 'wb').write(r.content)
