from abc import abstractmethod


class DataProvider:
    """
    DataProvider class.

    Contains different methods to load data and related metadata in a simple way.
    Each data source provides an implementation of this abstract base class.
    """

    @abstractmethod
    def get_series(self, ticker, start_index=None, end_index=None, group_by=None, group_mode=None):
        """
        Download data from data source
        """
        raise NotImplementedError
