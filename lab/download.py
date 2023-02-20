from etl.ree.downloader import REEDataDownloader

downloader = REEDataDownloader("demand", "demanda-tiempo-real")
downloader.download()
