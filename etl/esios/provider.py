import pandas as pd
from dateutil.parser import isoparse

from etl.provider import DataProvider
from etl.metadata import Metadata


class ESIOSDataProvider(DataProvider):

    def __init__(self):
        self.path = "datalake/clean/esios/"
        self.metadata = Metadata(self.path)
        self.tickers = self.metadata.get_tickers()
        self.ts_df = pd.read_csv(self.path+self.tickers[0]+".csv", index_col="DATE")
        for ticker in self.tickers[1:]:
            ts = pd.read_csv(self.path+ticker+".csv", index_col="DATE")
            self.ts_df = pd.concat([self.ts_df, ts], axis=1)

    def get_tickers(self):
        return self.tickers

    def get_metadata(self, ticker=None):
        return self.metadata.get_metadata(ticker)

    def get_series(self, ticker, freq=None, start_index=None, end_index=None, resample_by=None, group_mode=None):
        ts = self.ts_df[ticker].squeeze()
        if freq is not None:
            ts.index = pd.PeriodIndex(ts.index, freq=freq)
        ts = ts.sort_index()

        if start_index is not None:
            ts = ts.loc[ts.index >= start_index]
        if end_index is not None:
            ts = ts.loc[ts.index <= end_index]
        if resample_by is not None and group_mode is not None:
            ts = ts.resample(resample_by)
            if group_mode == "sum":
                ts = ts.sum()
            if group_mode == "mean":
                ts = ts.mean()

        return ts

    def get_all_series(self, freq=None, start_index=None, end_index=None, resample_by=None, group_mode=None):
        df = self.ts_df
        if freq is not None:
            df.index = pd.PeriodIndex(df.index, freq=freq)
        df = df.sort_index()

        if start_index is not None:
            df = df.loc[df.index >= start_index]
        if end_index is not None:
            df = df.loc[df.index <= end_index]
        if resample_by is not None and group_mode is not None:
            df = df.resample(resample_by)
            if group_mode == "sum":
                df = df.sum()
            if group_mode == "mean":
                df = df.mean()

        return df


if __name__ == '__main__':
    prov = ESIOSDataProvider()
    print(prov.get_tickers())
    print(prov.get_metadata("DEMANDA_REAL"))
    print(prov.get_series("DEMANDA_REAL"))

    start_index = isoparse("2022-12-01 00:00:00+01:00")
    end_index = isoparse("2022-12-31 00:00:00+01:00")
    print(prov.get_series("DEMANDA_REAL", start_index=start_index))
    print(prov.get_series("DEMANDA_REAL", start_index=start_index, end_index=end_index))

    print(prov.get_series("DEMANDA_REAL", resample_by="W", group_mode="mean"))

    print(prov.get_series("PRECIO_MERCADO_SPOT_DIARIO", start_index=start_index, end_index=end_index))