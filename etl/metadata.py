import pandas as pd
import xml.etree.ElementTree as etree
from dateutil.parser import isoparse


class Metadata:

    def __init__(self, path):
        self.path = path
        xml = etree.parse(self.path + "/schema.xml")
        root = xml.getroot()

        columns = ["ticker", "description", "category", "frequency", "region", "since", "until", "source"]
        self.metadata = pd.DataFrame(columns=columns)

        for node in root:
            ticker = node.find("ticker").text
            description = node.find("description").text
            category = node.find("category").text
            frequency = node.find("frequency").text
            region = node.find("region").text
            since = isoparse(node.find("since").text)
            until = isoparse(node.find("until").text)
            source = node.find("provider/source").text
            series = pd.Series([ticker, description, category, frequency, region, since, until, source], index=columns)
            self.metadata = pd.concat([self.metadata, series.to_frame().T], ignore_index=True)

    def get_tickers(self):
        return list(self.metadata["ticker"])

    def get_metadata(self):
        return self.metadata

    def get_metadata_ticker(self, ticker):
        return self.metadata.loc[self.metadata['ticker'] == ticker]


if __name__ == '__main__':
    md = Metadata("datalake/clean/esios")
    print(md.get_tickers())
    print(md.get_metadata())
    print(md.get_metadata_ticker("DEMANDA_REAL"))
