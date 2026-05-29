from __future__ import annotations

import math
import sys
import os
import threading
import time
from statistics import mean ,stdev

import numpy as np
import pytest

sys .path .insert (0 ,os .path .dirname (os .path .dirname (__file__ )))

from feature_store .rolling_engine import (
EntityRingBuffer ,
FeatureSnapshot ,
RollingFeatureEngine ,
haversine_km ,
)


def push_n (buf :EntityRingBuffer ,amounts :list [float ],offset_seconds :float =0.0 )->None :
    """Push a list of amounts with timestamps spaced 0.1 s apart."""
    now =time .time ()
    for i ,amt in enumerate (amounts ):
        ts =now -offset_seconds +i *0.1
        buf .push (ts ,amt ,40.0 ,-74.0 ,"test_merch")


class TestEntityRingBuffer :

    def test_empty_buffer_returns_zero_count (self ):
        buf =EntityRingBuffer (capacity =100 )
        stats =buf .window_stats (60 )
        assert stats .count ==0
        assert stats .total ==0.0

    def test_single_push_reflected_in_stats (self ):
        buf =EntityRingBuffer (capacity =100 )
        buf .push (time .time (),42.5 ,0 ,0 ,"test")
        stats =buf .window_stats (10 )
        assert stats .count ==1
        assert abs (stats .total -42.5 )<1e-9
        assert abs (stats .mean -42.5 )<1e-9
        assert stats .std ==0.0

    def test_window_excludes_old_events (self ):
        buf =EntityRingBuffer (capacity =100 )
        old_ts =time .time ()-10
        buf .push (old_ts ,999.0 ,0 ,0 ,"old")
        buf .push (time .time (),10.0 ,0 ,0 ,"new")

        stats_5s =buf .window_stats (5 )
        stats_60s =buf .window_stats (60 )

        assert stats_5s .count ==1 ,"Old event must be excluded from 5s window"
        assert stats_60s .count ==2 ,"Old event must be included in 60s window"

    def test_ring_buffer_wrap_around (self ):
        """Writes that wrap the ring buffer must still produce correct stats."""
        cap =10
        buf =EntityRingBuffer (capacity =cap )
        amounts =list (range (1 ,cap *2 +1 ))
        for amt in amounts :
            buf .push (time .time (),float (amt ),0 ,0 ,"m")

        stats =buf .window_stats (60 )

        assert stats .count ==cap
        expected_sum =sum (amounts [-cap :])
        assert abs (stats .total -expected_sum )<1e-6

    def test_std_calculation (self ):
        buf =EntityRingBuffer (capacity =100 )
        amounts =[10.0 ,20.0 ,30.0 ]
        for amt in amounts :
            buf .push (time .time (),amt ,0 ,0 ,"m")
        stats =buf .window_stats (60 )

        expected_std =float (np .std (amounts ))
        assert abs (stats .std -expected_std )<1e-6

    def test_unique_merchants (self ):
        buf =EntityRingBuffer (capacity =100 )
        for merchant in ["alpha","beta","alpha","gamma","beta"]:
            buf .push (time .time (),10.0 ,0 ,0 ,merchant )
        assert buf .unique_merchants (60 )==3

    def test_thread_safety (self ):
        """Concurrent pushes must not corrupt the buffer."""
        buf =EntityRingBuffer (capacity =500 )
        errors =[]

        def _writer (start_amt :float )->None :
            try :
                for i in range (100 ):
                    buf .push (time .time (),start_amt +i ,0 ,0 ,"m")
                    time .sleep (0.0001 )
            except Exception as exc :
                errors .append (exc )

        threads =[threading .Thread (target =_writer ,args =(i *1000 ,))for i in range (5 )]
        for t in threads :
            t .start ()
        for t in threads :
            t .join ()

        assert not errors ,f"Thread errors: {errors }"
        stats =buf .window_stats (60 )
        assert stats .count >0

    def test_last_location_updates (self ):
        buf =EntityRingBuffer (capacity =100 )
        buf .push (time .time (),10.0 ,51.5 ,-0.1 ,"m")
        buf .push (time .time (),20.0 ,48.8 ,2.3 ,"m")
        lat ,lon =buf .last_location ()
        assert abs (lat -48.8 )<1e-6
        assert abs (lon -2.3 )<1e-6


class TestHaversine :

    def test_same_point_is_zero (self ):
        assert haversine_km (40.7 ,-74.0 ,40.7 ,-74.0 )==0.0

    def test_nyc_to_london_approx (self ):
        dist =haversine_km (40.7128 ,-74.0060 ,51.5074 ,-0.1278 )
        assert 5500 <dist <5600 ,f"Expected ~5570 km, got {dist :.1f}"

    def test_zero_origin_returns_zero (self ):

        assert haversine_km (0.0 ,0.0 ,40.7 ,-74.0 )==0.0

    def test_symmetry (self ):
        d1 =haversine_km (40.7 ,-74.0 ,51.5 ,-0.1 )
        d2 =haversine_km (51.5 ,-0.1 ,40.7 ,-74.0 )
        assert abs (d1 -d2 )<1e-6


class TestRollingFeatureEngine :

    def test_new_entity_returns_zero_features (self ):
        engine =RollingFeatureEngine ()
        snap =engine .compute_features ("unknown_entity",current_amount =100.0 )
        assert snap .txn_count_1s ==0
        assert snap .txn_count_60s ==0
        assert snap .amount_zscore ==0.0

    def test_ingest_then_compute (self ):
        engine =RollingFeatureEngine ()
        engine .ingest ("alice",50.0 ,40.7 ,-74.0 ,"grocery")
        engine .ingest ("alice",75.0 ,40.7 ,-74.0 ,"grocery")

        snap =engine .compute_features ("alice",current_amount =60.0 )
        assert snap .txn_count_60s ==2
        assert abs (snap .amount_sum_60s -125.0 )<1e-6
        assert abs (snap .amount_mean_60s -62.5 )<1e-6

    def test_window_isolation (self ):
        """Events older than 1s should NOT appear in the 1s window."""
        engine =RollingFeatureEngine ()
        old_ts =time .time ()-5.0
        engine .ingest ("bob",999.0 ,0 ,0 ,"m",ts =old_ts )
        engine .ingest ("bob",10.0 ,0 ,0 ,"m")

        snap =engine .compute_features ("bob")

        assert snap .txn_count_1s ==1
        assert snap .txn_count_5s ==1
        assert snap .txn_count_60s ==2

    def test_zscore_anomaly (self ):
        """A very large amount should yield a high z-score."""
        engine =RollingFeatureEngine ()

        rng =np .random .default_rng (42 )
        for amt in rng .normal (loc =20.0 ,scale =2.0 ,size =20 ):
            engine .ingest ("carol",float (amt ),0 ,0 ,"grocery")

        snap =engine .compute_features ("carol",current_amount =2000.0 )
        assert snap .amount_zscore >5.0 ,(
        f"Expected z-score >> 5 for extreme amount, got {snap .amount_zscore }"
        )

    def test_velocity_ratio_increases_with_burst (self ):
        """Burst of transactions should show elevated velocity ratio."""
        engine =RollingFeatureEngine ()

        for _ in range (5 ):
            engine .ingest ("dave",50.0 ,0 ,0 ,"m",ts =time .time ()-30 )


        for _ in range (5 ):
            engine .ingest ("dave",50.0 ,0 ,0 ,"m")

        snap =engine .compute_features ("dave")
        assert snap .velocity_change_ratio >1.0

    def test_geo_distance_computed (self ):
        """After two transactions at different locations, geo_distance > 0."""
        engine =RollingFeatureEngine ()
        engine .ingest ("eve",100.0 ,40.7 ,-74.0 ,"atm")
        snap =engine .compute_features ("eve",current_lat =51.5 ,current_lon =-0.1 )
        assert snap .geo_distance_delta >5000

    def test_unique_merchants_60s (self ):
        engine =RollingFeatureEngine ()
        for m in ["shop_a","shop_b","shop_a","shop_c"]:
            engine .ingest ("frank",10.0 ,0 ,0 ,m )
        snap =engine .compute_features ("frank")
        assert snap .unique_merchants_60s ==3

    def test_model_array_length (self ):
        engine =RollingFeatureEngine ()
        engine .ingest ("grace",55.0 ,0 ,0 ,"m")
        snap =engine .compute_features ("grace",current_amount =55.0 )
        arr =snap .to_model_array ()
        assert arr .shape ==(20 ,),f"Expected 20 features, got {arr .shape }"
        assert arr .dtype ==np .float32

    def test_max_entities_eviction (self ):
        engine =RollingFeatureEngine (max_entities =10 )
        for i in range (15 ):
            engine .ingest (f"entity_{i }",10.0 ,0 ,0 ,"m")
        assert len (engine ._buffers )<=10

    def test_stats_counter (self ):
        engine =RollingFeatureEngine ()
        engine .ingest ("henry",1.0 ,0 ,0 ,"m")
        engine .ingest ("henry",2.0 ,0 ,0 ,"m")
        engine .compute_features ("henry")
        s =engine .stats ()
        assert s ["total_events_ingested"]==2
        assert s ["total_features_computed"]==1


class TestFeatureSnapshot :

    def test_to_dict_round_trip (self ):
        snap =FeatureSnapshot (entity_id ="x",computed_at =1234.0 ,current_amount =99.0 )
        d =snap .to_dict ()
        assert d ["entity_id"]=="x"
        assert d ["current_amount"]==99.0
        assert "amount_zscore"in d

    def test_to_model_array_no_nans (self ):
        snap =FeatureSnapshot (entity_id ="y",computed_at =0.0 )
        arr =snap .to_model_array ()
        assert not np .any (np .isnan (arr ))
        assert not np .any (np .isinf (arr ))


class TestConcurrency :

    def test_concurrent_ingest_and_read (self ):
        """
        Many threads writing and reading simultaneously must not crash
        or produce corrupt data.
        """
        engine =RollingFeatureEngine (buffer_capacity =200 )
        errors =[]

        def _writer (entity :str )->None :
            try :
                for i in range (200 ):
                    engine .ingest (entity ,float (i %50 +1 ),40.7 ,-74.0 ,"m")
                    time .sleep (0.0 )
            except Exception as exc :
                errors .append (("writer",exc ))

        def _reader (entity :str )->None :
            try :
                for _ in range (100 ):
                    snap =engine .compute_features (entity )
                    arr =snap .to_model_array ()
                    assert not np .any (np .isnan (arr ))
                    time .sleep (0.0 )
            except Exception as exc :
                errors .append (("reader",exc ))

        entities =[f"stress_{i }"for i in range (5 )]
        threads =[]
        for e in entities :
            threads .append (threading .Thread (target =_writer ,args =(e ,)))
            threads .append (threading .Thread (target =_reader ,args =(e ,)))

        for t in threads :
            t .start ()
        for t in threads :
            t .join ()

        assert not errors ,f"Concurrent errors: {errors }"
