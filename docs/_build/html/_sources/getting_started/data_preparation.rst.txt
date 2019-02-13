Download data
=============

Firstly, you should export your clickstream data as csv or other table
format (or you can download data directly from `BigQuery <bigquery.md>`__).

Data should have at least three columns: ``user_id``,
``event_timestamp`` and ``event_name``.

Prepare data for analysis
=========================

First of all, load the data in python using pandas:

.. code:: python

    import pandas as pd
    data = pd.read_csv('path_to_your_data.csv')

You also could read data from other sources such as ``.xlsx``
(``pd.read_excel``), ``sql`` (``pd.read_sql``) and etc. Please, check
the `pandas
documentation <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`__
for other options

Columns renaming and formatting
-------------------------------

Analysis submodule needs proper names of columns:

1. Column with user ID should be named as ``user_pseudo_id``
2. Name of event should be named as ``event_name``
3. Timestamp of event should be named as ``event_timestamp``. Also, it is needed
   to convert it to the integer type (seconds from ``1970-01-01``).

Rename your columns with pandas:

.. code:: python

    data = data.rename({
        'your_user_id_column_name': 'user_pseudo_id',
        'your_event_name_column_name': 'event_name',
        'your_event_timestamp_name': 'event_timestamp'
    }, axis=1)

Check the type of your timestamp column:

.. code:: python

    print("""
    Event timestamp type: {}
    Event timestamp example: {}
    """.format(
        data.event_timestamp.dtype,
        data.event_timestamp.iloc[0]
    ))

Out:

.. code:: none

    Event timestamp type: obj
    Event timestamp example: 2019-02-09 16:10:23

We see that here column with the timestamp is a python object (string).

You can use the following functions to convert it into seconds:

.. code:: python

    # converts string to datetime 
    data.event_timestamp = pd.to_datetime(data.event_timestamp)

    # converts datetime to integer
    data.event_timestamp = data.event_timestamp.astype(int) / 1e6

Add target events
-----------------

Most of our tools aim to estimate how different trajectories leads to
different target events. So you should add such events as
``lost`` and ``passed``.

For example, there is a list of events that correspond to the passed onboarding:

.. code:: python

    from retentioneering import preparing
    event_filter = ['newFlight', 'feed', 'tabbar', 'myFlights']
    data = preparing.add_passed_event(data, positive_event_name='passed', filter=event_filter)

And all users who were not passed over some time have lost event:

.. code:: python

    data = preparing.add_lost_event(data, existed_event='pass', time_thresh=5)

Export data
-----------

.. code:: python

    data.to_csv('prepared_data.csv', index=False)

