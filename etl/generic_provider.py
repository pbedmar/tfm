import pandas as pd
from dateutil.parser import isoparse

from etl.provider import DataProvider
from etl.metadata import Metadata


class GenericDataProvider(DataProvider):

    def __init__(self, path, want_metadata=False, tickers=[]):
        self.path = path
        self.want_metadata = want_metadata
        self.tickers = tickers

        if want_metadata:
            self.metadata = Metadata(self.path)

    def get_tickers(self):
        if self.want_metadata:
            return self.metadata.get_tickers()
        elif len(self.tickers) != 0:
            return self.tickers
        else:
            raise NotImplementedError

    def get_metadata(self, ticker=None):
        if self.want_metadata:
            return self.metadata.get_metadata(ticker)
        else:
            return NotImplementedError

    def get_series(self, ticker, start_index=None, end_index=None, resample_by=None, group_mode=None):
        ts = pd.read_csv(self.path+ticker+".csv", index_col="DATE").squeeze()
        ts.index = pd.PeriodIndex(ts.index, freq="M")
        ts = ts.sort_index()
        # ts.index = pd.to_datetime(ts.index).to_pydatetime()
        # ts = ts.asfreq("MS")
        # ts.index = ts.index.to_period("M")

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


if __name__ == '__main__':
    # omie = GenericDataProvider("datalake/clean/omie/", want_metadata=False, tickers=["PRECIO_OMIE"])
    # print(omie.get_tickers())
    # print(omie.get_series("PRECIO_OMIE"))
    #
    # start_index = isoparse("2022-12-01 00:00:00+01:00")
    # end_index = isoparse("2022-12-31 00:00:00+00:00")
    # print(omie.get_series("PRECIO_OMIE", start_index=start_index))
    # print(omie.get_series("PRECIO_OMIE", start_index=start_index, end_index=end_index))
    #
    # print(omie.get_series("PRECIO_OMIE", resample_by="W", group_mode="mean"))

    investingdotcom = GenericDataProvider("datalake/clean/investingdotcom/", want_metadata=False, tickers=["MONTHLY_TTF_PRICE"])
    print(investingdotcom.get_series("MONTHLY_TTF_PRICE"))
