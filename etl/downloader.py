from abc import abstractmethod


class DataDownloader:
    """
    DataDownloader class.

    Contains different methods to download and organize data from different sources.
    Each data source provides an implementation of this abstract base class.
    """

    @abstractmethod
    def download(self):
        """
        Download data from data source
        """
        raise NotImplementedError

    @abstractmethod
    def etl(self):
        """
        Extract, transform, and load downloaded data into project data repository
        """
        raise NotImplementedError
