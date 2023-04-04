import os
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta
from urllib.error import HTTPError

import pandas as pd
import shutil
from xml.dom import minidom
import xml.etree.ElementTree as etree

from etl.esios.ESIOS import ESIOS
from etl.downloader import DataDownloader


class ESIOSDataDownloader(DataDownloader):
    """
    ESIOSDataDownloader class

    DataDownloader implementation for ESIOS.
    """

    def __init__(self, indicators, start_date, end_date, path="datalake/"):
        self.path = path
        self.source = "ESIOS"
        self.token = "fa264a1e89e02bdb141e216f4c53c3ad53574be997e4150cd4a439bcce5207a6"
        self.file_format = "ESIOS wrapper by random user"
        self.start_date = start_date
        self.end_date = end_date
        self.indicators = indicators

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

    def xml_provider(self):
        provider_xml = etree.Element("provider")

        source_xml = etree.SubElement(provider_xml, "source")
        source_xml.text = self.source

        format_xml = etree.SubElement(provider_xml, "format")
        format_xml.text = self.file_format

        url_xml = etree.SubElement(provider_xml, "url")
        url_xml.text = "https://api.esios.ree.es"

        return provider_xml

    def download(self):
        esios = ESIOS(self.token)
        names = esios.get_names(self.indicators)
        print("DOWNLOADER -- ESIOSDataDownloader.download() for", names)

        for indicator, name in zip(self.indicators, names):
            partial_dfs = []
            start_date_local = self.start_date
            print("Indicator:", name)
            while start_date_local < self.end_date:
                end_date_local = start_date_local + relativedelta(days=15) + relativedelta(minutes=-1)
                if end_date_local > self.end_date:
                    end_date_local = self.end_date

                print("Downloading from", start_date_local, "to", end_date_local)

                result = None
                while result is None:
                    try:
                        result = esios.get_data(indicator, start_date_local, end_date_local)
                    except HTTPError as err:
                        print(" ERROR -- Error in ESIOSDataDownloader.download() -- server returned", err.code,
                              "accessing url")
                    if result is None:
                        print("No data available")

                partial_dfs.append(result[["value", "geo_name"]])
                start_date_local = start_date_local + relativedelta(days=15)

            clean_name = name.replace(" ", "_").upper()
            df = pd.concat(partial_dfs)
            df = df.reset_index(level=0)
            df = df.rename(columns={"datetime_utc": "DATE", "value": clean_name})

            os.makedirs(os.path.dirname(self.raw_directory()), exist_ok=True)
            df.to_csv(self.raw_directory() + clean_name + ".csv", index=False)

    def add_xml_variable(self, ticker, description, category, frequency, region, since, until):
        var_xml = etree.Element("variable")

        ticker_xml = etree.SubElement(var_xml, "ticker")
        ticker_xml.text = ticker

        description_xml = etree.SubElement(var_xml, "description")
        description_xml.text = description

        category_xml = etree.SubElement(var_xml, "category")
        category_xml.text = category

        frequency_xml = etree.SubElement(var_xml, "frequency")
        frequency_xml.text = frequency

        region_xml = etree.SubElement(var_xml, "region")
        region_xml.text = region

        since_xml = etree.SubElement(var_xml, "since")
        since_xml.text = since.astimezone().isoformat()

        until_xml = etree.SubElement(var_xml, "until")
        until_xml.text = until.astimezone().isoformat()

        provider_xml = self.xml_provider()
        var_xml.append(provider_xml)

        return var_xml

    def etl(self):
        os.makedirs(os.path.dirname(self.clean_directory()), exist_ok=True)
        xml_file = open(self.metadata_filename(), "w", encoding="utf8")
        root_xml = etree.Element("variables")

        ticker = "DEMANDA_REAL"
        description = "Es el valor real de la demanda de energía eléctrica medida en tiempo real."
        category = "energy / demand / real"
        frequency = "every 10 minutes - grouped (mean) by hour"
        region = "peninsula"
        var_xml = self.add_xml_variable(ticker, description, category, frequency, region, self.start_date, self.end_date)
        root_xml.append(var_xml)
        shutil.copy(self.raw_filename(ticker), self.clean_filename(ticker))
        df = pd.read_csv(self.raw_filename(ticker), index_col="DATE")
        df.index = pd.to_datetime(df.index)
        df = df[[ticker]].resample("H").mean()
        df.to_csv(self.clean_filename(ticker))

        ticker = "GENERACIÓN_MEDIDA_TOTAL"
        description = "Medidas de la generación según el tipo de producción utilizado. El desglose de este indicador " \
                      "proporciona las medidas de generación de los distintos tipos de producción, de consumo bombeo " \
                      "y la generación en el enlace de Baleares. Este indicador se puede desglosar por provincias."
        category = "energy / generation / measured"
        frequency = "hour"
        region = "peninsula"
        var_xml = self.add_xml_variable(ticker, description, category, frequency, region, self.start_date, self.end_date)
        root_xml.append(var_xml)
        df = pd.read_csv(self.raw_filename(ticker))
        df = df.groupby(df['DATE']).sum()  # doesn't contain data about Balears and Canary Islands
        df.index = pd.to_datetime(df.index)
        df.to_csv(self.clean_filename(ticker))

        tickers = ["GENERACIÓN_MEDIDA_EÓLICA_TERRESTRE", "GENERACIÓN_MEDIDA_CICLO_COMBINADO",
                   "GENERACIÓN_MEDIDA_DERIVADOS_DEL_PETRÓLEO_Ó_CARBÓN", "GENERACIÓN_MEDIDA_GAS_NATURAL_COGENERACIÓN",
                   "GENERACIÓN_MEDIDA_HIDRÁULICA", "GENERACIÓN_MEDIDA_NUCLEAR",
                   "GENERACIÓN_MEDIDA_SOLAR_FOTOVOLTAICA"]
        description = " "
        category = "energy / generation / measured"
        frequency = "hour"
        for ticker in tickers:
            region = "peninsula"
            var_xml = self.add_xml_variable(ticker, description, category, frequency, region, self.start_date, self.end_date)
            root_xml.append(var_xml)
            df = pd.read_csv(self.raw_filename(ticker))
            df.index = pd.to_datetime(df.index)
            df = df.groupby(df['DATE']).sum()  # doesn't contain data about Balears and Canary Islands
            df.index = pd.to_datetime(df.index)
            df.to_csv(self.clean_filename(ticker))

        ticker = "PRECIO_MERCADO_SPOT_DIARIO"
        description = "El Mercado Diario es un mercado mayorista en el que se establecen transacciones de energía " \
                      "eléctrica para el día siguiente, mediante la presentación de ofertas de venta y adquisición de " \
                      "energía eléctrica por parte de los participantes en el mercado. Como resultado del mismo se " \
                      "determina de forma simultánea el precio del mercado diario en cada zona de oferta, " \
                      "los programas de toma y entrega de energía, y los programas de intercambio entre zonas de " \
                      "oferta. En concreto este indicador se refiere al precio resultante del acoplamiento de los " \
                      "mercados diarios europeos (SDAC por sus siglas en inglés) en algunas de las zonas más " \
                      "representativas del mismo: MIBEL (España y Portugal), Francia, Bélgica, Países Bajos, " \
                      "Italia y Alemania. Este indicador está georeferenciado a nivel de país. Las principales " \
                      "fuentes de información son la Plataforma de Transparencia de ENTSO-E y el Operador del Mercado " \
                      "Ibérico, Polo Español (OMIE). Se presenta también, el precio resultante del mercado gestionado " \
                      "por N2EX en el Reino Unido (externo al acoplamiento de los mercados diarios europeos), " \
                      "calculado en €/MWh con la información disponible (precio en GBP/MWh y ratio de conversión " \
                      "€/GBP) en la web de NordPool."
        category = "energy / price / spot"
        frequency = "hour"
        region = "spain"
        var_xml = self.add_xml_variable(ticker, description, category, frequency, region, self.start_date, self.end_date)
        root_xml.append(var_xml)
        df = pd.read_csv(self.raw_filename(ticker), index_col="DATE")
        df = df.loc[df['geo_name'] == "España"]
        df = df[[ticker]]
        df.to_csv(self.clean_filename(ticker))

        xmlstr = minidom.parseString(etree.tostring(root_xml)).toprettyxml(indent="    ")
        xml_file.write(xmlstr)


if __name__ == '__main__':
    # 1293 -> demanda real
    # 10043 -> generacion medida total
    # 1159 -> Generación medida Eólica terrestre
    # 1156 -> Generación medida ciclo combinado
    # 1165 -> Generación medida Derivados del petróleo ó carbón
    # 1164 -> Generación medida Gas Natural Cogeneración
    # 10035 -> Generación medida Hidráulica
    # 1153 -> Generación medida Nuclear
    # 1161 -> Generación medida Solar fotovoltaica
    # 600 -> PRECIO MERCADO SPOT DIARIO

    indicators = [1293, 10043, 1159, 1156, 1165, 1164, 10035, 1153, 1161, 600]
    start_date = isoparse("2014-01-01T00:00+00:00")
    end_date = isoparse("2022-12-31T23:59+00:00")

    downloader = ESIOSDataDownloader(indicators, start_date, end_date)

    downloader.download()
    downloader.etl()
