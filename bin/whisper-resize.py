#!/usr/bin/env python

import os
import sys
import math
import time
import bisect
import signal
import optparse
import traceback

try:
  import whisper
except ImportError:
  raise SystemExit('[ERROR] Please make sure whisper is installed properly')

class Approximator(object):
  "Generates approximated value for arbitrary timestamp in time-series"
  def __init__(self, series=[]):
    self.datapoints = []
    self.timestamps = []
    self.values = []
    for datapoints in series:
      self.loadDatapoints(datapoints)

  def loadDatapoints(self, datapoints):
    self.datapoints.extend(filter( lambda x: x[1] is not None, datapoints))
    self.datapoints.sort(cmp=lambda a,b: cmp(a[0], b[0]))
    self.timestamps = [d[0] for d in self.datapoints]
    self.values = [d[1] for d in self.datapoints]

  def linearValue(self, timestamp, max_gap=None):
    "Linear approximation based on two neighbor datapoints"
    left_timestamp_id = bisect.bisect(self.timestamps, timestamp) - 1
    if left_timestamp_id < 0 or left_timestamp_id >= len(self.timestamps):
      return None
    left_timestamp = self.timestamps[left_timestamp_id]
    left_value = float(self.values[left_timestamp_id])
    if left_timestamp == timestamp:
      return left_value
    right_timestamp_id = left_timestamp_id + 1
    if right_timestamp_id >= len(self.timestamps):
      return None
    right_value = float(self.values[right_timestamp_id])
    right_timestamp = self.timestamps[right_timestamp_id]
    # Preserve gaps on a graph
    if min((timestamp-left_timestamp),(right_timestamp-timestamp)) > max_gap:
        return None
    k = (right_value - left_value) / (right_timestamp - left_timestamp)
    v = left_value + k * (timestamp - left_timestamp)
    return v

# Ignore SIGPIPE
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

now = int(time.time())

option_parser = optparse.OptionParser(
    usage='''%prog path timePerPoint:timeToStore [timePerPoint:timeToStore]*

timePerPoint and timeToStore specify lengths of time, for example:

60:1440      60 seconds per datapoint, 1440 datapoints = 1 day of retention
15m:8        15 minutes per datapoint, 8 datapoints = 2 hours of retention
1h:7d        1 hour per datapoint, 7 days of retention
12h:2y       12 hours per datapoint, 2 years of retention
''')

option_parser.add_option(
    '--xFilesFactor', default=None,
    type='float', help="Change the xFilesFactor")
option_parser.add_option(
    '--aggregationMethod', default=None,
    type='string', help="Change the aggregation function (%s)" %
    ', '.join(whisper.aggregationMethods))
option_parser.add_option(
    '--force', default=False, action='store_true',
    help="Perform a destructive change")
option_parser.add_option(
    '--newfile', default=None, action='store',
    help="Create a new database file without removing the existing one")
option_parser.add_option(
    '--nobackup', action='store_true',
    help='Delete the .bak file after successful execution')
option_parser.add_option(
    '--aggregate', action='store_true',
    help='Try to aggregate the values to fit the new archive better.'
         ' Note that this will make things slower and use more memory.')
option_parser.add_option(
    '--approximate', action='store_true',
    help='If new interval is wider than original resulting time series will be sparse.'
         ' This option fills in missing datapoints based on existing series approximation.')

(options, args) = option_parser.parse_args()

if len(args) < 2:
  option_parser.print_help()
  sys.exit(1)

path = args[0]

if not os.path.exists(path):
  sys.stderr.write("[ERROR] File '%s' does not exist!\n\n" % path)
  option_parser.print_help()
  sys.exit(1)

info = whisper.info(path)

new_archives = [whisper.parseRetentionDef(retentionDef)
                for retentionDef in args[1:]]

old_archives = info['archives']
# sort by precision, lowest to highest
old_archives.sort(key=lambda a: a['secondsPerPoint'], reverse=True)

if options.xFilesFactor is None:
  xff = info['xFilesFactor']
else:
  xff = options.xFilesFactor

if options.aggregationMethod is None:
  aggregationMethod = info['aggregationMethod']
else:
  aggregationMethod = options.aggregationMethod

print('Retrieving all data from the archives')
for archive in old_archives:
  fromTime = now - archive['retention'] + archive['secondsPerPoint']
  untilTime = now
  timeinfo,values = whisper.fetch(path, fromTime, untilTime)
  archive['data'] = (timeinfo,values)

if options.newfile is None:
  tmpfile = path + '.tmp'
  if os.path.exists(tmpfile):
    print('Removing previous temporary database file: %s' % tmpfile)
    os.unlink(tmpfile)
  newfile = tmpfile
else:
  newfile = options.newfile

print('Creating new whisper database: %s' % newfile)
whisper.create(newfile, new_archives, xFilesFactor=xff, aggregationMethod=aggregationMethod)
size = os.stat(newfile).st_size
print('Created: %s (%d bytes)' % (newfile,size))

if options.aggregate:
  # This is where data will be interpolated (best effort)
  print('Migrating data with aggregation...')
  all_datapoints = []
  for archive in old_archives:
    # Loading all datapoints into memory for fast querying
    timeinfo, values = archive['data']
    new_datapoints = zip( range(*timeinfo), values )
    if all_datapoints:
      last_timestamp = all_datapoints[-1][0]
      slice_end = 0
      for i,(timestamp,value) in enumerate(new_datapoints):
        if timestamp > last_timestamp:
          slice_end = i
          break
      all_datapoints += new_datapoints[i:]
    else:
      all_datapoints += new_datapoints

  oldtimestamps = map( lambda p: p[0], all_datapoints)
  oldvalues = map( lambda p: p[1], all_datapoints)

  print("oldtimestamps: %s" % oldtimestamps)
  # Simply cleaning up some used memory
  del all_datapoints

  new_info = whisper.info(newfile)
  new_archives = new_info['archives']

  for archive in new_archives:
    step = archive['secondsPerPoint']
    fromTime = now - archive['retention'] + now % step
    untilTime = now + now % step + step
    print("(%s,%s,%s)" % (fromTime,untilTime, step))
    timepoints_to_update = range(fromTime, untilTime, step)
    print("timepoints_to_update: %s" % timepoints_to_update)
    newdatapoints = []
    for tinterval in zip( timepoints_to_update[:-1], timepoints_to_update[1:] ):
      # TODO: Setting lo= parameter for 'lefti' based on righti from previous
      #       iteration. Obviously, this can only be done if
      #       timepoints_to_update is always updated. Is it?
      lefti = bisect.bisect_left(oldtimestamps, tinterval[0])
      righti = bisect.bisect_left(oldtimestamps, tinterval[1], lo=lefti)
      newvalues = oldvalues[lefti:righti]
      if newvalues:
        non_none = filter( lambda x: x is not None, newvalues)
        if 1.0*len(non_none)/len(newvalues) >= xff:
          newdatapoints.append([tinterval[0],
                                whisper.aggregate(aggregationMethod,
                                                  non_none, newvalues)])
    whisper.update_many(newfile, newdatapoints)
elif options.approximate:
  print 'Migrating data with approximation...'
  approximator = Approximator()
  for archive in old_archives:
    timeinfo, values = archive['data']
    datapoints = zip( range(*timeinfo), values )
    approximator.loadDatapoints(datapoints)
  new_info = whisper.info(newfile)
  new_archives = new_info['archives']
  for archive in new_archives:
    step = archive['secondsPerPoint']
    fromTime = now - archive['retention'] + now % step
    untilTime = now + now % step + step
    new_datapoints = []
    for ts in xrange(fromTime, untilTime, step):
      val = approximator.linearValue(ts, max_gap=step*2)
      if val is not None:
        new_datapoints.append((ts, val))
    whisper.update_many(newfile, new_datapoints)
else:
  print('Migrating data without aggregation...')
  for archive in old_archives:
    timeinfo, values = archive['data']
    datapoints = zip( range(*timeinfo), values )
    datapoints = filter(lambda p: p[1] is not None, datapoints)
    whisper.update_many(newfile, datapoints)

if options.newfile is not None:
  sys.exit(0)

backup = path + '.bak'
print('Renaming old database to: %s' % backup)
os.rename(path, backup)

try:
  print('Renaming new database to: %s' % path)
  os.rename(tmpfile, path)
except:
  traceback.print_exc()
  print('\nOperation failed, restoring backup')
  os.rename(backup, path)
  sys.exit(1)

if options.nobackup:
  print("Unlinking backup: %s" % backup)
  os.unlink(backup)
