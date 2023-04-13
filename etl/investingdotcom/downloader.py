import os

from dateutil.parser import isoparse
import pandas as pd

from etl.downloader import DataDownloader


class InvestingDotComDataDownloader(DataDownloader):
    """
    InvestingDotComDataDownloader class

    DataDownloader implementation for investing.com.
    """

    def __init__(self, path="datalake/"):
        self.path = path
        self.source = "investingdotcom"
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

        # DAILY TTF PRICE
        start_date = isoparse("2017-10-23")
        end_date = isoparse("2023-04-11")
        date_range = pd.date_range(start_date, end_date, freq="D")

        ticker = "DAILY_TTF_PRICE"
        description = "TTF gas index daily prices."
        category = "gas / price"
        frequency = "daily"
        region = "Netherlands"

        df = pd.read_csv(self.raw_filename("daily_ttf"), index_col=0)
        df = df.drop(['Open', 'High', "Low", "Vol.", "Change %"], axis=1)

        df.index = pd.to_datetime(df.index)
        df = df.reindex(date_range)
        df = df.fillna(method="ffill")

        print("\nLongitud del dataset:", len(df))
        print("Número de fechas que deben existir en ese rango:", len(date_range))
        print("Número de valores nulos en el dataset:\n", df.isna().sum())

        df.index.names = ["DATE"]
        df.columns = [ticker]
        df.to_csv(self.clean_filename(ticker))

        # MENSUAL TTF PRICE
        start_date = isoparse("2010-04-01")
        end_date = isoparse("2023-04-01")
        date_range = pd.date_range(start_date, end_date, freq="MS")

        ticker = "MONTHLY_TTF_PRICE"
        description = "TTF gas index monthly prices."
        category = "gas / price"
        frequency = "monthly"
        region = "Netherlands"

        df = pd.read_csv(self.raw_filename("monthly_ttf"), index_col=0)
        df = df.drop(['Open', 'High', "Low", "Vol.", "Change %"], axis=1)
        df.index = pd.to_datetime(df.index)

        print("\nLongitud del dataset:", len(df))
        print("Número de fechas que deben existir en ese rango:", len(date_range))
        print("Número de valores nulos en el dataset:\n", df.isna().sum())

        df.index.names = ["DATE"]
        df.columns = [ticker]
        df.to_csv(self.clean_filename(ticker))


if __name__ == '__main__':
    downloader = InvestingDotComDataDownloader()
    downloader.etl()
