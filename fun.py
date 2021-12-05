import os
import requests
import re

def downloadcsv():
    reg = r"hw4/[\w]*.csv"
    urls = ['https://sapienza2021adm.s3.eu-south-1.amazonaws.com/hw4/echonest.csv',
            'https://sapienza2021adm.s3.eu-south-1.amazonaws.com/hw4/features.csv',
            'https://sapienza2021adm.s3.eu-south-1.amazonaws.com/hw4/tracks.csv']
    for url in urls:
        response = requests.get(url)
        url_content = response.content
        os.walk('../data/csv/')
        csv_f = open('../data/csv/'+re.findall(reg, url)[0][4:], 'wb')
        csv_f.write(url_content)
        csv_f.close()
