import csv
import requests
import re

def downloadcsv():
    reg = r"hw4/[\w]*.csv"
    urls = ['https://sapienza2021adm.s3.eu-south-1.amazonaws.com/hw4/echonest.csv',
            'https://sapienza2021adm.s3.eu-south-1.amazonaws.com/hw4/features.csv',
            'https://sapienza2021adm.s3.eu-south-1.amazonaws.com/hw4/tracks.csv']
    for url in urls:
        response = requests.get(url)
        with open('data/csv/'+re.findall(reg, url)[0][4:], 'w') as f:
            writer = csv.writer(f)
            for line in response.iter_lines():
                writer.writerow(line.decode('utf-8').split(','))
if __name__ == "__main__":
    downloadcsv()