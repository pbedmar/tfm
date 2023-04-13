import os

from dateutil.parser import isoparse
import pandas as pd

from etl.downloader import DataDownloader


class MarketwatchDataDownloader(DataDownloader):
    """
    MarketwatchDataDownloader class

    DataDownloader implementation for marketwatch.com.
    """

    def __init__(self, path="datalake/"):
        self.path = path
        self.source = "marketwatch"
        self.file_format = "multiple CSVs"
        self.start_date = isoparse("2010-12-17")
        self.end_date = isoparse("2023-03-31")

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

        start_date = isoparse("2010-12-17")
        end_date = isoparse("2023-03-31")
        date_range = pd.date_range(start_date, end_date, freq="D")

        ticker = "DAILY_COAL_PRICE"
        description = "ARGUS-McCloskey API 2 coal prices. The industry standard reference price for coal imported into northwest Europe."
        category = "coal / price"
        frequency = "daily"
        region = "Northwest Europe"

        df = pd.read_csv(self.raw_filename("coal_"+str(start_date.year)), index_col=0)
        df = df.drop(['High', "Low", "Close"], axis=1)
        for year in range(start_date.year+1, end_date.year+1):
            df_year = pd.read_csv(self.raw_filename("coal_" + str(year)), index_col=0)
            df_year = df_year.drop(['High', "Low", "Close"], axis=1)
            df = pd.concat([df, df_year])

        df.index = pd.to_datetime(df.index)
        df = df.reindex(date_range)
        df = df.fillna(method="ffill")

        print("\nLongitud del dataset:", len(df))
        print("Número de fechas que deben existir en ese rango:", len(date_range))
        print("Número de valores nulos en el dataset:\n", df.isna().sum())

        df.index.names = ["DATE"]
        df.columns = [ticker]
        df.to_csv(self.clean_filename(ticker))


if __name__ == '__main__':
    downloader = MarketwatchDataDownloader()
    downloader.etl()