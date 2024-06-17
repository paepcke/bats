'''
Created on May 20, 2024

@author: paepcke

Apart from the unit tests, this test file also 
creates a .csv file with a three hour time period
at 1 second granularity.

The dataframe that can load from the .csv file includes
a column called 'data', which is 1 during every quarter 
hour minute. All other times the value is 0.

So, during minutes 0, 15, 30, and 45 the 'data' value
is 1, else 0.

The files/dataframe also holds two columns: minutes_sin, 
and minutes_cos, which can be used to capture the cyclical
nature of the content, if included in a trained model. 
'''
from data_calcs.utils import (
    TimeGranularity,
    Utils,
    PDJson)
from datetime import (
    datetime,
    timedelta)
from pathlib import (
    Path)
import io
import json
import os
import pandas as pd
import tempfile
import unittest


#**********TEST_ALL = True
TEST_ALL = False

class UtilsTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a cyclical datafile 
        # in <this-dir>/tmp_tst_data/three_minutes.csv:
        cls.make_three_hr_cookoo()

    def setUp(self):
        self.make_test_data()


    def tearDown(self):
        self.tmpdir.cleanup()

    # ------------------------- Test Routines ----------------
    
    def test_cyclical_time_encoding(self):
        
        # Start with 10 time ticks of 1 minute each
        start_date = datetime(2024, 1, 1, 15, 0, 0)
        #****end_date   = datetime(2024, 1, 1, 15, 10, 0)
        end_date   = datetime(2024, 1, 1, 15, 2, 0)
        
        
        dateseries = pd.date_range(start_date, end_date, freq=timedelta(seconds=1), normalize=False, inclusive='left')
        encoding   = Utils.cyclical_time_encoding(dateseries, TimeGranularity.MINUTES)
        pd.testing.assert_frame_equal(encoding, self.two_min_df)

        # Try zero registration at second 15, though the
        # time date only has 2 minutes:
        with self.assertRaises(ValueError):
            encoding   = Utils.cyclical_time_encoding(dateseries, 
                                                      TimeGranularity.MINUTES,
                                                      zero_registration=15)
        
        # Something with two periods:
        end_date   = datetime(2024, 1, 1, 15, 30, 0)
        dateseries = pd.date_range(start_date, end_date, freq=timedelta(seconds=1), normalize=False, inclusive='left')
        encoding   = Utils.cyclical_time_encoding(dateseries, TimeGranularity.MINUTES)

        self.assertEqual(encoding.minutes_sin[0], 0.0)
        self.assertEqual(encoding.minutes_cos[0], 1.0)
        
        self.assertEqual(encoding.minutes_sin[900], 0.0)
        self.assertEqual(encoding.minutes_cos[900], 1.0)     

        self.assertEqual(encoding.minutes_sin.iloc[-1], 0.0)
        self.assertEqual(encoding.minutes_cos.iloc[-1], 1.0)     
        
        
        # Now for real:
        end_date   = datetime(2024, 1, 1, 15, 30, 0)
        dateseries = pd.date_range(start_date, end_date, freq=timedelta(seconds=1), normalize=False, inclusive='left')
        # Get the sin and cosine just before the reach 0 and 1, respectively:
        sin_penultimate = encoding.minutes_sin.iloc[-1] 
        cos_penultimate = encoding.minutes_cos.iloc[-1] 
        encoding   = Utils.cyclical_time_encoding(dateseries, 
                                                  TimeGranularity.MINUTES,
                                                  zero_registration=15)

        # Should have sin==0, and cos==1 at the first
        # 15 minute point. The 15 is seconds, therefore the *60
        self.assertEqual(encoding.minutes_sin[15*60], 0.0)
        self.assertEqual(encoding.minutes_cos[15*60], 1.0)     
        
        self.assertEqual(encoding.minutes_sin[15*60-1], sin_penultimate)
        self.assertEqual(encoding.minutes_cos[15*60-1], cos_penultimate)
        
        # Make values that turn on every 15 minutes:
        # 14*60 seconds of 0, then a 1 for a second:
        qrt_on_off_vals = [0]*14*60 + [1]*60
        qtr_hr = pd.Series(qrt_on_off_vals * (int(len(dateseries) / len(qrt_on_off_vals))))
        vals_df = pd.DataFrame({'date' : dateseries[:len(qtr_hr)],
                                'min_ticks' : qtr_hr
                               })
        data = pd.concat([vals_df, encoding], axis='columns')
        # We should see a 1 in the min_ticks column on the
        # quarter hour
        
        qtr_min_start = 14*60
        qtr_min_stop  = 15*60
        # At quarter hour edge:
        self.assertEqual(data.iloc[qtr_min_start]['min_ticks'], 1)
        # But one second earlier:
        self.assertEqual(data.iloc[qtr_min_start - 1]['min_ticks'], 0)
        # One second past the 15 minutes:
        self.assertEqual(data.iloc[qtr_min_stop]['min_ticks'], 0)
        # But the last second of the 15th minute:
        self.assertEqual(data.iloc[qtr_min_stop - 1]['min_ticks'], 1)
        
        # At the end of the half-hour minute:
        self.assertEqual(data.iloc[-1]['min_ticks'], 1)
        # At the start of the half-hour minute:
        self.assertEqual(data.iloc[-60]['min_ticks'], 1)
        # But the second before (the last sec of the 29th minute):
        self.assertEqual(data.iloc[-61]['min_ticks'], 0)

    #------------------------------------
    # test_cycle_time
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_cycle_time(self):
        
        # Seconds at 0
        dt = datetime(2020, 1, 1, 15, 20, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.SECONDS)
        self.assertEqual(sin, 0.0)
        self.assertEqual(cos, 1.0)
        
        # Seconds at 1/4 of a min: 15 seconds
        dt = datetime(2020, 1, 1, 15, 20, 15)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.SECONDS)
        self.assertEqual(sin, 1.0)
        self.assertEqual(round(cos, 5), 0.0)

        # Seconds at 1/2 of a min: 30 seconds
        dt = datetime(2020, 1, 1, 15, 20, 30)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.SECONDS)
        self.assertEqual(round(sin,4), 0.0)
        self.assertEqual(cos, -1.0)
        
        # Seconds at 3/4 of a min: 45 seconds
        dt = datetime(2020, 1, 1, 15, 20, 45)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.SECONDS)
        self.assertEqual(sin, -1.0)
        self.assertEqual(round(cos, 4), 0.0)

        # For the others, just do spot checks:
        
        # Minutes
        dt = datetime(2020, 1, 1, 15, 0, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.MINUTES)
        self.assertEqual(sin, 0.0)
        self.assertEqual(cos, 1.0)

        # 3/4 of an hour:
        dt = datetime(2020, 1, 1, 15, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.MINUTES)
        self.assertEqual(sin, -1.0)
        self.assertEqual(round(cos, 4), 0.0)
        
        # Hours:
        dt = datetime(2020, 1, 1, 0, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.HOURS)
        self.assertEqual(sin, 0.0)
        self.assertEqual(cos, 1.0)

        # 3/4 of a day:        
        dt = datetime(2020, 1, 1, 18, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.HOURS)
        self.assertEqual(sin, -1.0)
        self.assertEqual(round(cos, 4), 0.0)
        
        # Days: first day:
        dt = datetime(2020, 1, 1, 0, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.DAYS)
        self.assertEqual(round(sin, 4), 0.2079)
        self.assertEqual(round(cos, 4), 0.9781)
        
        # 3/4 of a month (approximately):
        dt = datetime(2020, 1, 23, 18, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.DAYS)
        self.assertEqual(round(sin, 4), -0.9945)
        self.assertEqual(round(cos, 4), 0.1045)

        # Months: January:
        dt = datetime(2020, 1, 10, 0, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.MONTHS)
        self.assertEqual(round(sin, 4), 0.5)
        self.assertEqual(round(cos, 4), 0.8660)
        
        # September
        dt = datetime(2020, 9, 23, 18, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.MONTHS)
        self.assertEqual(sin, -1.0)
        self.assertEqual(round(cos, 4), 0.0)

        # Years (Decades): decade boundary:
        dt = datetime(2020, 1, 10, 0, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.YEARS)
        self.assertEqual(round(sin, 4), 0.0)
        self.assertEqual(cos, 1.0)
        
        # 3/4 of a decade: 
        dt = datetime(2027, 9, 22, 18, 45, 0)
        (sin, cos) = Utils.cycle_time(dt, TimeGranularity.YEARS)
        self.assertEqual(round(sin, 4), -0.9511)
        self.assertEqual(round(cos, 4), -0.309)

        # Try getting sin/cos for one datetime
        # for each granularity:
        dt = datetime(2020, 1, 1, 15, 0, 0)
        sin_cos_dict = Utils.cycle_time(dt, None)
        
        # Round the values:
        for time_gran, sin_cos in sin_cos_dict.items():
            sin_cos_dict[time_gran] = (round(sin_cos[0], 4), 
                                       round(sin_cos[1], 4))

        expected = {TimeGranularity.SECONDS: (0.0, 1.0),
                    TimeGranularity.MINUTES: (0.0, 1.0),
                    TimeGranularity.HOURS  : (round(-0.7071067811865471, 4),    round(-0.7071067811865479, 4)),
                    TimeGranularity.DAYS   : (round(0.20791169081775931, 4),    round(0.9781476007338057, 4)),
                    TimeGranularity.MONTHS : (round(0.49999999999999994, 4),    round(0.8660254037844387, 4)),
                    TimeGranularity.YEARS  : (round(-4.898587196589413e-16, 4), 1.0)}

        self.assertDictEqual(sin_cos_dict, expected)

        
    #------------------------------------
    # test_find_file_by_timestamp
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_find_file_by_timestamp(self):
        
        
        self.search_dir
        time1 = Utils.extract_file_timestamp(self.search_fname1)
        
        # Just a timestamp of fname1, which is also the one forfname4;
        # no prefix or suffix filter:

        # Should retrieve fname1 and fname4: they have the
        # same timestamp, and we set latest to False:
        files = set(Utils.find_file_by_timestamp(self.search_dir, 
                                                 timestamp=time1,
                                                 latest=False
                                                 )) 
        expected = set([self.search_fname1, self.search_fname4])
        self.assertSetEqual(files, expected)

        # Now set latest to True, which should return
        # only one of the two files whose timestamp matches:
        files = Utils.find_file_by_timestamp(self.search_dir, 
                                             timestamp=time1,
                                             latest=True
                                             )
        # We expect one of these, but undefined
        # which one: 
        expected = set([self.search_fname1, self.search_fname4])
        self.assertEqual(len(files), 1)
        self.assertIn(files[0], expected)
        
        # Now, distinguish between fname1 and fname4 by adding
        # prefix filter:        
        files = set(Utils.find_file_by_timestamp(self.search_dir, 
                                                 timestamp=time1,
                                                 prefix='my_prefix' 
                                                 ))
        expected = set([self.search_fname1])
        self.assertSetEqual(files, expected)
        
        # Only filter by the suffix:
        files = set(Utils.find_file_by_timestamp(self.search_dir, 
                                                 timestamp=time1,
                                                 suffix='.txt' 
                                                 ))
        expected = set([self.search_fname4])
        self.assertSetEqual(files, expected)
        
    #------------------------------------
    # test_dt_timestamp_conversions 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dt_timestamp_conversions(self):
        
        str_stamp = '2023-02-14T14_23_40'
        dt = Utils.datetime_from_timestamp(str_stamp)
        expected = datetime(2023, 2, 14, 14,23, 40)
        self.assertEqual(dt, expected)
        
        # And back:
        str_stamp_recovered = Utils.timestamp_from_datetime(dt)
        self.assertEqual(str_stamp_recovered, str_stamp)
         
    #------------------------------------
    # test_extract_species_from_wav_filename
    #-------------------
    
    #********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_extract_species_from_wav_filename(self):
        
        s1 = 'barn1_D20220207T215546m654-Laci-Tabr.wav'        
        s2 = 'barn1_D20220207T214358m129-Coto.wav'
        s3 = 'barn1_D20220720T020517m043.wav'

        extract = Utils.extract_species_from_wav_filename(s1)
        expect  = ['Laci', 'Tabr']
        self.assertListEqual(extract,expect)

        extract = Utils.extract_species_from_wav_filename(s2)
        expect  = ['Coto']
        self.assertListEqual(extract,expect)

        extract = Utils.extract_species_from_wav_filename(s3)
        expect  = []
        self.assertListEqual(extract,expect)
         
    #------------------------------------
    # test_PDJson 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_PDJson(self):
        
        # pd.Series
        
        ser = pd.Series({'serkey1' : 'blue', 'serkey2' : 'green'}, name='colors')
        jstr = json.dumps(ser, cls=PDJson)
        expected = '"{\\"__pd.series__\\": \\"{\\\\\\"serkey1\\\\\\":\\\\\\"blue\\\\\\",\\\\\\"serkey2\\\\\\":\\\\\\"green\\\\\\"}\\", \\"name\\": \\"colors\\"}"'
        #****self.assertEqual(expected, jstr)
        
        recovered = json.loads(jstr, object_hook=PDJson.decode)
        #****** ERROR: AttributeError: 'str' object has no attribute 'dtype' 
        pd.testing.assert_series_equal(recovered, ser)
        
        my_dict = {'foo' : 10, 'bar' : ser}
        jstr = json.dumps(my_dict, cls=PDJson)
        
        expected = '{"foo": 10, "bar": "{\\"__pd.series__\\" : {\\"data\\" : {\\"serkey1\\":\\"blue\\",\\"serkey2\\":\\"green\\"}, \\"name\\" : \\"colors\\"}}"}'
        self.assertEqual(jstr, expected)

        recovered = json.loads(jstr, object_hook=PDJson.decode)
        self.assertEqual(len(recovered), len(my_dict))
        self.assertEqual(recovered['foo'], 10)
        pd.testing.assert_series_equal(recovered['bar'], ser)
        
        # pd.DataFrame
        df = pd.DataFrame({'foo' : [10, 20], 'bar' : [ser, ser]})
        jstr = json.dumps(df, cls=PDJson)
        
        expected = '"{\\"__pd.dataframe__\\" : {\\"data\\" : {\\"foo\\":{\\"0\\":10,\\"1\\":20},\\"bar\\":{\\"0\\":{\\"serkey1\\":\\"blue\\",\\"serkey2\\":\\"green\\"},\\"1\\":{\\"serkey1\\":\\"blue\\",\\"serkey2\\":\\"green\\"}}}}}"'
        self.assertEqual(jstr, expected)

        recovered = json.loads(jstr, object_hook=PDJson.decode)
        self.assertTrue(isinstance(recovered, pd.DataFrame))
        self.assertEqual(len(recovered), len(df))
        
        pd.testing.assert_frame_equal(recovered, df)
        
        
# -------------------------- Utilities ------------------------

    #------------------------------------
    # make_test_data
    #-------------------
    
    def make_test_data(self):

        # Df with 2 minutes of 1-minute cycyles:
        buf = io.StringIO('{"minutes_sin":{"0":0.0,"1":0.1045284633,"2":0.2079116908,"3":0.3090169944,"4":0.4067366431,"5":0.5,"6":0.5877852523,"7":0.6691306064,"8":0.7431448255,"9":0.8090169944,"10":0.8660254038,"11":0.9135454576,"12":0.9510565163,"13":0.9781476007,"14":0.9945218954,"15":1.0,"16":0.9945218954,"17":0.9781476007,"18":0.9510565163,"19":0.9135454576,"20":0.8660254038,"21":0.8090169944,"22":0.7431448255,"23":0.6691306064,"24":0.5877852523,"25":0.5,"26":0.4067366431,"27":0.3090169944,"28":0.2079116908,"29":0.1045284633,"30":5.665538898e-16,"31":-0.1045284633,"32":-0.2079116908,"33":-0.3090169944,"34":-0.4067366431,"35":-0.5,"36":-0.5877852523,"37":-0.6691306064,"38":-0.7431448255,"39":-0.8090169944,"40":-0.8660254038,"41":-0.9135454576,"42":-0.9510565163,"43":-0.9781476007,"44":-0.9945218954,"45":-1.0,"46":-0.9945218954,"47":-0.9781476007,"48":-0.9510565163,"49":-0.9135454576,"50":-0.8660254038,"51":-0.8090169944,"52":-0.7431448255,"53":-0.6691306064,"54":-0.5877852523,"55":-0.5,"56":-0.4067366431,"57":-0.3090169944,"58":-0.2079116908,"59":0.0,"60":0.0,"61":0.1045284633,"62":0.2079116908,"63":0.3090169944,"64":0.4067366431,"65":0.5,"66":0.5877852523,"67":0.6691306064,"68":0.7431448255,"69":0.8090169944,"70":0.8660254038,"71":0.9135454576,"72":0.9510565163,"73":0.9781476007,"74":0.9945218954,"75":1.0,"76":0.9945218954,"77":0.9781476007,"78":0.9510565163,"79":0.9135454576,"80":0.8660254038,"81":0.8090169944,"82":0.7431448255,"83":0.6691306064,"84":0.5877852523,"85":0.5,"86":0.4067366431,"87":0.3090169944,"88":0.2079116908,"89":0.1045284633,"90":5.665538898e-16,"91":-0.1045284633,"92":-0.2079116908,"93":-0.3090169944,"94":-0.4067366431,"95":-0.5,"96":-0.5877852523,"97":-0.6691306064,"98":-0.7431448255,"99":-0.8090169944,"100":-0.8660254038,"101":-0.9135454576,"102":-0.9510565163,"103":-0.9781476007,"104":-0.9945218954,"105":-1.0,"106":-0.9945218954,"107":-0.9781476007,"108":-0.9510565163,"109":-0.9135454576,"110":-0.8660254038,"111":-0.8090169944,"112":-0.7431448255,"113":-0.6691306064,"114":-0.5877852523,"115":-0.5,"116":-0.4067366431,"117":-0.3090169944,"118":-0.2079116908,"119":0.0},"minutes_cos":{"0":1.0,"1":0.9945218954,"2":0.9781476007,"3":0.9510565163,"4":0.9135454576,"5":0.8660254038,"6":0.8090169944,"7":0.7431448255,"8":0.6691306064,"9":0.5877852523,"10":0.5,"11":0.4067366431,"12":0.3090169944,"13":0.2079116908,"14":0.1045284633,"15":2.832769449e-16,"16":-0.1045284633,"17":-0.2079116908,"18":-0.3090169944,"19":-0.4067366431,"20":-0.5,"21":-0.5877852523,"22":-0.6691306064,"23":-0.7431448255,"24":-0.8090169944,"25":-0.8660254038,"26":-0.9135454576,"27":-0.9510565163,"28":-0.9781476007,"29":-0.9945218954,"30":-1.0,"31":-0.9945218954,"32":-0.9781476007,"33":-0.9510565163,"34":-0.9135454576,"35":-0.8660254038,"36":-0.8090169944,"37":-0.7431448255,"38":-0.6691306064,"39":-0.5877852523,"40":-0.5,"41":-0.4067366431,"42":-0.3090169944,"43":-0.2079116908,"44":-0.1045284633,"45":-1.836970199e-16,"46":0.1045284633,"47":0.2079116908,"48":0.3090169944,"49":0.4067366431,"50":0.5,"51":0.5877852523,"52":0.6691306064,"53":0.7431448255,"54":0.8090169944,"55":0.8660254038,"56":0.9135454576,"57":0.9510565163,"58":0.9781476007,"59":1.0,"60":1.0,"61":0.9945218954,"62":0.9781476007,"63":0.9510565163,"64":0.9135454576,"65":0.8660254038,"66":0.8090169944,"67":0.7431448255,"68":0.6691306064,"69":0.5877852523,"70":0.5,"71":0.4067366431,"72":0.3090169944,"73":0.2079116908,"74":0.1045284633,"75":2.832769449e-16,"76":-0.1045284633,"77":-0.2079116908,"78":-0.3090169944,"79":-0.4067366431,"80":-0.5,"81":-0.5877852523,"82":-0.6691306064,"83":-0.7431448255,"84":-0.8090169944,"85":-0.8660254038,"86":-0.9135454576,"87":-0.9510565163,"88":-0.9781476007,"89":-0.9945218954,"90":-1.0,"91":-0.9945218954,"92":-0.9781476007,"93":-0.9510565163,"94":-0.9135454576,"95":-0.8660254038,"96":-0.8090169944,"97":-0.7431448255,"98":-0.6691306064,"99":-0.5877852523,"100":-0.5,"101":-0.4067366431,"102":-0.3090169944,"103":-0.2079116908,"104":-0.1045284633,"105":-1.836970199e-16,"106":0.1045284633,"107":0.2079116908,"108":0.3090169944,"109":0.4067366431,"110":0.5,"111":0.5877852523,"112":0.6691306064,"113":0.7431448255,"114":0.8090169944,"115":0.8660254038,"116":0.9135454576,"117":0.9510565163,"118":0.9781476007,"119":1.0}}')         
        self.two_min_df = pd.read_json(buf)
        
        self.tmpdir = tempfile.TemporaryDirectory(dir='/tmp', prefix='utils_tests_', delete=False)
        self.search_dir = os.path.join(self.tmpdir.name, 'file_find_dir')
        os.makedirs(self.search_dir)
        
        # Put two timestamped files, and one non-timestamped into the dir.
        # Then add one file with the same timestamp as fname1:
        #
        # fname1 : my_prefix_<time>.csv
        # fname2 : <time+1day>.txt
        # fname3 : random_file
        # fname4 : other_prefix_<time-of-fname1>.txt
        
        fname1_time = Utils.file_timestamp()
        self.search_fname1 = os.path.join(self.search_dir, f"my_prefix_{fname1_time}.csv")
        # Make the second timestamp a bit later than the one in fname1.
        # The replace makes the timestamp work for datetime.isoformat():
        fname2_dt = datetime.fromisoformat(fname1_time.replace('_', ':'))
        fname2_dt += timedelta(days=1)
        fname2_time = fname2_dt.strftime('%Y-%m-%d_%H_%M_%S')
        
        self.search_fname2 = os.path.join(self.search_dir, f"{fname2_time}.txt")
        self.search_fname3 = os.path.join(self.search_dir, 'random_file')
        # fname4 will be same timestamp as fname1:
        timestamp_fname1 = Utils.extract_file_timestamp(self.search_fname1)
        self.search_fname4 = os.path.join(self.search_dir, f"other_prefix_{timestamp_fname1}.txt")
        
        Path(self.search_fname1).touch()
        Path(self.search_fname2).touch()
        Path(self.search_fname3).touch()
        Path(self.search_fname4).touch()

    #------------------------------------
    # make_three_hr_cookoo
    #-------------------
    
    @staticmethod
    def make_three_hr_cookoo():
        # Start with 10 time ticks of 1 minute each
        start_date = datetime(2024, 1, 1, 15, 0, 0)
        #****end_date   = datetime(2024, 1, 1, 15, 10, 0)
        end_date   = datetime(2024, 1, 1, 18, 0, 0)
               
        dateseries = pd.date_range(start_date, 
                                   end_date, 
                                   freq=timedelta(seconds=1), 
                                   normalize=False, 
                                   inclusive='left')
        encoding   = Utils.cyclical_time_encoding(dateseries, TimeGranularity.MINUTES)
        
        qrt_on_off_vals = [1]*60 + [0]*14*60
        qtr_hr = pd.Series(qrt_on_off_vals * (int(len(dateseries) / len(qrt_on_off_vals))))
        vals_df = pd.DataFrame({'date' : dateseries[:len(qtr_hr)],
                                'data' : qtr_hr
                               })
        
        
        # The 0th min should be 1s:
        assert(vals_df.iloc[0].data == 1)
        assert(vals_df.iloc[0 + 59].data == 1)
        assert(vals_df.iloc[0 + 59 + 1].data == 0)
        
        # The 15th min should be 1s:
        assert(vals_df.iloc[15*60 - 1].data == 0)
        assert(vals_df.iloc[15*60].data == 1)
        assert(vals_df.iloc[15*60 + 59].data == 1)
        assert(vals_df.iloc[15*60 + 59 + 1].data == 0)
        
        # As should the 30th:
        assert(vals_df.iloc[30*60 - 1].data == 0)
        assert(vals_df.iloc[30*60].data == 1)
        assert(vals_df.iloc[30*60 + 59].data == 1)
        assert(vals_df.iloc[30*60 + 59 + 1].data == 0)
        
        # 45th:
        assert(vals_df.iloc[45*60 - 1].data == 0)
        assert(vals_df.iloc[45*60].data == 1)
        assert(vals_df.iloc[45*60 + 59].data == 1)
        assert(vals_df.iloc[45*60 + 59 + 1].data == 0)
        
        data = pd.concat([vals_df, encoding], axis='columns')
        tst_data_dir = os.path.join(os.path.dirname(__file__), 'tmp_tst_data')
        os.makedirs(tst_data_dir, exist_ok=True)
        data.index.name = 'tick_number'
        data_fname = 'three_minutes.csv'
        data.to_csv(os.path.join(tst_data_dir, data_fname))


# -------------------------- Main ------------------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()