import os
import requests
from datetime import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

import glob
import json
import pandas as pd

from etl.downloader import DataDownloader


class REEDataDownloader(DataDownloader):
    """
    REEDataDownloader class

    DataDownloader implementation for Red Eléctrica de España (REE).
    """

    def __init__(self, categories_widgets, path="datalake/"):
        self.path = path
        self.source = "ree"
        self.categories_widgets = categories_widgets
        self.url_prefix = f"https://apidatos.ree.es/es/datos/"

    def raw_directory(self, category, widget):
        return f"{self.path}raw/{self.source}/{category}/{widget}/"

    def raw_json_filename(self, category, widget, freq, start_date, end_date):
        return f"{self.raw_directory(category, widget)}{freq}_{start_date}_{end_date}.json"

    def build_url(self, category, widget, start_date, end_date, freq):
        return self.url_prefix + f"{category}/{widget}?start_date={start_date}&end_date={end_date}&time_trunc={freq}"

    def download(self, start_date, end_date, freq="day"):

        for category, widgets in self.categories_widgets.items():
            for widget in widgets:
                while start_date < end_date:
                    end_date_local = start_date + relativedelta(months=1) + relativedelta(minutes=-1)
                    if end_date_local > end_date:
                        end_date_local = end_date

                    start_date_s = start_date.isoformat(sep="T", timespec="minutes")
                    end_date_local_s = end_date_local.isoformat(sep="T", timespec="minutes")
                    url = self.build_url(category, widget, start_date_s, end_date_local_s, freq)

                    while True:
                        resp = requests.get(url)
                        if resp.ok:
                            break
                        else:
                            print("Error in REEDataDownloader.download() -- server returned", resp.reason, "accessing", url)

                    os.makedirs(os.path.dirname(self.raw_json_filename(category, widget, freq,
                                                                       start_date_s.replace(":", "H"),
                                                                       end_date_local_s).replace(":", "H")), exist_ok=True)
                    output = open(self.raw_json_filename(category, widget, freq,
                                                         start_date_s.replace(":", "H"),
                                                         end_date_local_s).replace(":", "H"), "wb")
                    output.write(resp.content)
                    output.close()

                    start_date = start_date + relativedelta(months=1)

    def etl(self):
        for category, widgets in self.categories_widgets.items():
            for widget in widgets:
                filename_list = sorted(glob.glob(self.raw_directory(category, widget) + "/*.json"))

                print(category, widget)
                print(filename_list)

                df_list = []

                for filename in filename_list:
                    # Load JSON as pandas dataframe
                    file = json.load(open(filename))
                    df_list.append(pd.DataFrame(file["included"]))

                print(df_list[0])


                #raw_df = pd.concat(df_list, ignore_index=True)


if __name__ == '__main__':
    # downloader = REEDataDownloader("demanda", "demanda-tiempo-real")
    # ree_categories_widgets = {"demanda": ["evolucion"],
    #                           "generacion": ["estructura-generacion"],
    #                           }
    ree_categorias_widgets = {"balance": ["balance-electrico"]}
    downloader = REEDataDownloader(ree_categorias_widgets)

    start_date_ = parse("2018-01-01T00:00")
    end_date_ = parse("2018-07-31T23:59")
    #downloader.download(start_date_, end_date_)
    downloader.etl()
