import os

from dateutil.parser import isoparse
import pandas as pd
import numpy as np

from etl.downloader import DataDownloader


class ICAPDataDownloader(DataDownloader):
    """
    ICAPDataDownloader class

    DataDownloader implementation for International Carbon Action Partnership.
    """

    def __init__(self, path="datalake/"):
        self.path = path
        self.source = "ICAP"
        self.file_format = "CSV"

    def raw_directory(self):
        return f"{self.path}raw/{self.source}/"

    def clean_directory(self):
        return f"{self.path}clean/{self.source}/"

    def raw_filename(self, name):
        return f"{self.raw_directory()}{name}.csv"

    def clean_filename(self, name):
        return f"{self.clean_directory()}{name}.csv"

    def metadata_filename(self):
        return f"{self.clean_directory()}schema.xml"

    def download(self):
        raise NotImplementedError

    def etl(self):
        os.makedirs(os.path.dirname(self.clean_directory()), exist_ok=True)
        # xml_file = open(self.metadata_filename(), "w", encoding="utf8")
        # root_xml = etree.Element("variables")

        # C02 EUA PRICE
        start_date = isoparse("2005-03-09")
        end_date = isoparse("2023-03-31")
        date_range = pd.date_range(start_date, end_date, freq="D")

        ticker = "CO2_EUA_PRICE"
        description = "EU Allowance climate credit used in the European Union Emissions Trading Scheme (EU ETS)"
        category = "co2 / price"
        frequency = "daily"
        region = "European Union"

        df = pd.read_csv(self.raw_filename("icap-co2-data"), index_col=0, sep=";", decimal=",")

        df.index = pd.to_datetime(df.index, format="%d.%m.%Y")
        df = df.reindex(date_range)
        df = df.replace(-1, np.nan)
        df = df.fillna(method="ffill")

        print("\nLongitud del dataset:", len(df))
        print("Número de fechas que deben existir en ese rango:", len(date_range))
        print("Número de valores nulos en el dataset:\n", df.isna().sum())

        df.index.names = ["DATE"]
        df.columns = [ticker]
        df.to_csv(self.clean_filename(ticker))


if __name__ == '__main__':
    downloader = ICAPDataDownloader()
    downloader.etl()
