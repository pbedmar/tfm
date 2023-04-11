import os

from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from urllib.error import HTTPError

import pandas as pd
import numpy as np
import shutil
from xml.dom import minidom
import xml.etree.ElementTree as etree

from etl.downloader import DataDownloader


class OMIEDataDownloader(DataDownloader):
    """
    ESIOSDataDownloader class

    DataDownloader implementation for ESIOS.
    """

    def __init__(self, path="datalake/"):
        self.path = path
        self.source = "OMIE"
        self.file_format = "XLSX"
        self.start_date = isoparse("1998-01-01T00:00+01:00")
        self.end_date = isoparse("2023-02-28T23:00+01:00")

    def raw_directory(self):
        return f"{self.path}raw/{self.source}/"

    def clean_directory(self):
        return f"{self.path}clean/{self.source}/"

    def raw_filename(self, name):
        return f"{self.raw_directory()}{name}.xlsx"

    def clean_filename(self, name):
        return f"{self.clean_directory()}{name}.csv"

    def metadata_filename(self):
        return f"{self.clean_directory()}schema.xml"

    def download(self):
        raise NotImplementedError

    def etl(self):
        os.makedirs(os.path.dirname(self.clean_directory()), exist_ok=True)
        xml_file = open(self.metadata_filename(), "w", encoding="utf8")
        root_xml = etree.Element("variables")
        date_range = pd.date_range(self.start_date, self.end_date, freq="H")

        ticker = "PRECIO_OMIE"
        description = "Precios de la electricidad obtenidos desde OMIE."
        category = "energy / price"
        frequency = "hourly data"
        region = "Spain"
        years = range(self.start_date.year, self.end_date.year+1)

        xls = pd.ExcelFile(self.raw_filename("prices"))
        df = pd.read_excel(xls, str(years[0]))
        for year in years[1:]:
            df_year = pd.read_excel(xls, str(year))
            df = pd.concat([df, df_year])

        # rename columns
        new_column_names = ["DATE"] + [str(i) for i in range(0, 25)]
        df.columns = new_column_names
        df = df.drop("24", axis=1)

        # NAs interpolation
        print("Número de valores nulos en el dataset:\n", df.isna().sum())
        df["2"] = df["2"].interpolate()
        df["23"] = df["23"].interpolate()
        print("\nNúmero de valores nulos en el dataset tras la limpieza:\n", df.isna().sum())
        print(df["DATE"].dtypes)

        # arrange hours in an horizontal fashion
        df = pd.melt(df, id_vars=['DATE'], var_name='hour', value_name='value')
        df['DATE'] = df['DATE'].dt.strftime('%Y-%m-%d') + ' ' + df['hour'] + ':00:00'
        df = df.set_index(pd.to_datetime(df['DATE']))
        df = df.drop(['hour', 'DATE'], axis=1)
        df = df.sort_index()

        print("\nLongitud del dataset:", len(df))
        print("Número de fechas que deben existir en ese rango", len(date_range))

        df.to_csv(self.clean_filename(ticker))

if __name__ == '__main__':
    downloader = OMIEDataDownloader()
    downloader.etl()
