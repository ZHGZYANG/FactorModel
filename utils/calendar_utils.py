import pandas as pd
import numpy as np
import datetime

from config import root_fldr

class Calendar_Util(object):
  def __init__(self):
    self._data = pd.read_pickle(f"{root_fldr}/reference/calendar.pkl")
    self._data = self._data[['calendar_date', 'is_open']]
    self._data['calendar_date'] = pd.to_datetime(self._data['calendar_date']).apply(lambda x: x.date())
    self._data = self._data.sort_values('calendar_date').reset_index(drop=True)
    self._data['idx'] = np.cumsum(self._data.is_open)

  def dateWrap(self, datadate, by=0):
    if datadate.__class__.__name__ == "Timestamp":
      datadate = datadate.date()

    if not self.isbizday(datadate):
      by = by + 1

    self._inRangeCheck(datadate)

    startidx = int(self._data[self._data['calendar_date'] == datadate].idx)
    endidx = startidx + by

    res = self._data[(self._data.idx == endidx) & (self._data.is_open == 1)]

    if len(res) == 0:
      raise Exception("%s (%d) out of range" % (datadate, by))

    res = res['calendar_date'].values[0]
    return pd.to_datetime(res)

  def fastdateWrap(self, datadates, by=0):
    if type(datadates) == pd.Series:
      datadates = datadates.tolist()
    elif type(datadates) == list:
      pass
    else:
      raise Exception('datadates should be only pd.series or list of pd.datatime')

    data = pd.DataFrame({'calendar_date': datadates})

    mapping = self._data[self._data.is_open == 1].reset_index(drop=True)
    mapping['calendar_date'] = pd.to_datetime(mapping['calendar_date'])
    data = pd.merge(data, mapping, on='calendar_date', how='left')

    if any(np.isnan(data.idx)):
      holiday = data[np.isnan(data.idx)].loc[0, 'calendar_date']
      raise Exception("datadates has holiday %s" % (holiday.date()))

    data['idx_tgt'] = data['idx'] + by
    mappingtgt = mapping[['calendar_date', 'idx']]
    mappingtgt = mappingtgt.rename(columns={'idx': 'idx_tgt', 'calendar_date': 'date_tgt'})

    data = pd.merge(data, mappingtgt, on=['idx_tgt'], how='left')

    return data['date_tgt'].tolist()

  def _inRangeCheck(self, datadate):
    if datadate.__class__.__name__ == "Timestamp":
      datadate = datadate.date()

    if (datadate > max(self._data.calendar_date)) or (datadate < min(self._data.calendar_date)):
      raise Exception("%s out range" % (datadate))

  def isbizday(self, datadate):
    if datadate.__class__.__name__ == "Timestamp":
      datadate = datadate.date()

    self._inRangeCheck(datadate)

    return all(self._data[self._data['calendar_date'] == datadate].is_open)

  def dateSeq(self, startdate, enddate, alldates=False):
    if startdate.__class__.__name__ == "Timestamp":
      startdate = startdate.date()
    if enddate.__class__.__name__ == "Timestamp":
      enddate = enddate.date()

    self._inRangeCheck(startdate)
    self._inRangeCheck(enddate)

    if alldates:
      res = self._data[(self._data.calendar_date >= startdate) & (self._data.calendar_date <= enddate)]
    else:
      res = self._data[(self._data.calendar_date >= startdate) & (self._data.calendar_date <= enddate) & (self._data.is_open == 1)]

    res = res.calendar_date.tolist()

    res = [pd.to_datetime(d) for d in res]
    return res

  def startofmonth(self, datadate):
    return pd.to_datetime(datadate.strftime("%Y%m01"))

  def startofweek(self, datadate):
    return datadate - datetime.timedelta(days=datadate.weekday())
