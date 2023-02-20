import os
import sys
import requests

from etl.downloader import DataDownloader


class REEDataDownloader(DataDownloader):
    """
    REEDataDownloader class

    DataDownloader implementation for Red Eléctrica de España (REE).
    """

    def __init__(self, category, widget, path="datalake/"):
        self.path = path
        self.source = "ree"
        self.category = category
        self.widget = widget
        self.raw_filename = "data.json"
        self.url_prefix = f"https://apidatos.ree.es/es/datos/{category}/{widget}"

    def raw_json_filename(self):
        return self.path + "raw/" + self.source + "/" + self.category + "/" + self.widget + "/" + self.raw_filename

    def build_url(self, start_date, end_date, time_trunc):
        return self.url_prefix + f"?start_date={start_date}&end_date={end_date}&time_trunc={time_trunc}"

    def download(self, start_date="2018-01-01T00:00", end_date="2022-12-31T23:59", time_trunc="hour"):
        url = self.build_url(start_date, end_date, time_trunc)
        resp = requests.get(url)
        print(self.raw_json_filename())
        print(os.path.dirname(self.raw_json_filename()))
        os.makedirs(os.path.dirname(self.raw_json_filename()), exist_ok=True)
        output = open(self.raw_json_filename(), "wb")
        output.write(resp.content)
        output.close()



class EurostatPriceDataDownloader(DataDownloader):
    '''
    EurostatPricesDataDownloader class

    DataDownloader implementation for EuroStat prices.
    '''

    def __init__(self, path="datalake/"):
        self.path = path
        self.source = "ree"
        self.file_format = "XLSX Oil Market Prices"
        self.input_filename = "olive-oil-market-prices_en_0.xlsx"
        self.url = "https://ec.europa.eu/info/sites/default/files/food-farming-fisheries/plants_and_plant_products/" \
                   "documents/olive-oil-market-prices_en_0.xlsx"

    def excel_filename(self):
        return self.path + "raw/" + self.source + "/" + self.input_filename

    def metadata_filename(self, interval='W'):
        if interval == 'W':
            return self.path + "data/" + self.source + "/schema.xml"
        else:
            return self.path + "data/" + self.source + "/schema." + interval + ".xml"

    def csv_filename(self, ticker):
        return self.path + "data/" + self.source + "/" + ticker + ".csv"

    def download(self):
        # Downloading excel file from source
        resp = requests.get(self.url)
        os.makedirs(os.path.dirname(self.excel_filename()), exist_ok=True)
        output = open(self.excel_filename(), "wb")
        output.write(resp.content)
        output.close()

    interval_prefix = {"W": "", "M": "MONTHLY_"}
    interval_frequency = {"W": "weekly", "M": "monthly"}

    def group(self, df, interval="W"):
        '''
        Group data by date interval 
        (W: week, M: month end, MS: month start, Q: quarter end, QS: quarter start, Y: year end, YS: year start)
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        '''
        sub_df = df.groupby(pd.Grouper(freq=interval, key="ReferenceTo")).agg({"Price": "mean"}).round(decimals=2)
        sub_df = sub_df.reset_index()
        return sub_df

    def etl(self, interval="W"):
        # Opening excel file with pandas
        dataframe = pd.read_excel(self.excel_filename(), sheet_name="Data", engine="openpyxl")

        # Only keep the interesting columns
        indexing_variables = ["Category", "Member State", "Market", "ReferenceTo", "Price"]
        dataframe = dataframe[indexing_variables]

        # Expand columns that generate a subset of the dataframe
        categories = pd.unique(dataframe["Category"])
        member_states = pd.unique(dataframe["Member State"])
        markets = pd.unique(dataframe["Market"])

        os.makedirs(os.path.dirname(self.metadata_filename()), exist_ok=True)
        xml_file = open(self.metadata_filename(interval), "w", encoding="utf8")
        root_xml = ET.Element("variables")

        # Loop over each possible column combination
        for category in categories:
            for member_state in member_states:
                for market in markets:
                    # Subset generation
                    sub_dataframe = dataframe[
                        (dataframe["Category"] == category) & (dataframe["Member State"] == member_state)
                        & (dataframe["Market"] == market)]

                    # We only store store the subset if it's not empty
                    if not sub_dataframe.empty:

                        ticker = self.interval_prefix[interval] + (
                                    category.upper() + "_" + member_state.upper() + "_" + market.upper()).replace(" ",
                                                                                                                  "")

                        sys.stdout.reconfigure(encoding='utf-8')
                        print(ticker)

                        output_filename = self.csv_filename(ticker)

                        # Only keep date & price columns. The subset will be sorted by date.
                        sub_dataframe = sub_dataframe[["ReferenceTo", "Price"]]

                        if interval != 'W':  # Group data
                            sub_dataframe = self.group(sub_dataframe, interval)

                        sub_dataframe = sub_dataframe.rename(
                            columns={"ReferenceTo": "DATE", "Price": ticker}).sort_values(by=["DATE"])

                        # Store as CSV
                        sub_dataframe.to_csv(output_filename, index=False, header=True, encoding="utf8")

                        # Generating variable corpus to be pasted in the xml schema file
                        var_xml = ET.Element("variable")
                        root_xml.append(var_xml)

                        ticker_xml = ET.SubElement(var_xml, "ticker")
                        ticker_xml.text = ticker

                        description_xml = ET.SubElement(var_xml, "description")
                        description_xml.text = "Precios del aceite de oliva en " + market + " (" + member_state + ")"

                        category_xml = ET.SubElement(var_xml, "category")
                        category_xml.text = "price"

                        frequency_xml = ET.SubElement(var_xml, "frequency")
                        frequency_xml.text = self.interval_frequency[interval]

                        since_xml = ET.SubElement(var_xml, "since")
                        since_xml.text = sub_dataframe.iloc[0, 0].strftime('%Y-%m-%d')

                        provider_xml = ET.Element("provider")
                        var_xml.append(provider_xml)

                        source_xml = ET.SubElement(provider_xml, "source")
                        source_xml.text = self.source

                        format_xml = ET.SubElement(provider_xml, "format")
                        format_xml.text = self.file_format

                        url_xml = ET.SubElement(provider_xml, "url")
                        url_xml.text = self.url

                        country_xml = ET.SubElement(provider_xml, "country")
                        country_xml.text = member_state

                        category_prov_xml = ET.SubElement(provider_xml, "category")
                        category_prov_xml.text = category

                        market_xml = ET.SubElement(provider_xml, "market")
                        market_xml.text = market

        # Lint and write on xml file
        xmlstr = minidom.parseString(ET.tostring(root_xml)).toprettyxml(indent="    ")
        xml_file.write(xmlstr)

        xml_file.close()


if __name__ == '__main__':
    downloader = EurostatPriceDataDownloader()
    print("Downloading Eurostat prices...")
    # downloader.download()
    print("Extracting data from Excel file...")
    downloader.etl()
    downloader.etl(interval='M')
