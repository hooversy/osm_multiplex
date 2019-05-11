"""
.. moduleauthor:: Sylvan Hoover <hooversy@oregonstate.edu>
"""

# third-party libraries
import tables
import pandas as pd

def anomaly_detection(data):
    """
    Generate report of probabilities of anomalous data collection

    Parameters
    ----------
    data1 : 
        Dataset either containing just the first counter data or data
        from both counting systems

    data2 : list
        Data from the second counting system if not part of data1 file

    Returns
    -------
    report : 
        Probabilities of anomalous data collection
    """
    data_store = pd.HDFStore('data.h5',mode='w')
    for chunk in read_csv(data1,chunksize=50000):
        data_store.append('rider_counts',chunk)
    data_store.close()

    lstm_results = lstm(data_store.rider_counts)

def lstm(data):
    print('Under development')