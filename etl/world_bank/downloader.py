import os

from dateutil.parser import isoparse
import pandas as pd

from etl.downloader import DataDownloader


class WorldBankDataDownloader(DataDownloader):
    """
    WorldBankDataDownloader class

    DataDownloader implementation for World Bank.
    """

    def __init__(self, path="datalake/"):
        self.path = path
        self.source = "world_bank"
        self.file_format = "XLS"
        self.start_date = isoparse("1960")
        self.end_date = isoparse("2021")

    def raw_directory(self):
        return f"{self.path}raw/{self.source}/"

    def clean_directory(self):
        return f"{self.path}clean/{self.source}/"

    def raw_filename(self, name):
        return f"{self.raw_directory()}{name}.xls"

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

        start_date = isoparse("1960")
        end_date = isoparse("2021")
        date_range = pd.date_range(start_date, end_date, freq="YS")

        ticker = "GDP"
        description = "ARGUS-McCloskey API 2 coal prices. The industry standard reference price for coal imported into northwest Europe."
        category = "macro / gdp"
        frequency = "yearly"
        region = "Spain"

        df = pd.read_excel(self.raw_filename("gdp"), sheet_name="Data")
        df = df.loc[df['Country Code'] == "ESP"]
        df = df.drop(["Country Name", "Country Code", "Indicator Name", "Indicator Code"], axis=1)
        df = df.T
        print(df)

        print("\nLongitud del dataset:", len(df))
        print("Número de fechas que deben existir en ese rango:", len(date_range))
        print("Número de valores nulos en el dataset:\n", df.isna().sum())

        df.index.names = ["DATE"]
        df.columns = [ticker]
        df.to_csv(self.clean_filename(ticker))


if __name__ == '__main__':
    downloader = WorldBankDataDownloader()
    downloader.etl()