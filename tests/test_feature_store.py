from __future__ import annotations

import sys
import os
import threading
import time

import pytest

sys .path .insert (0 ,os .path .dirname (os .path .dirname (__file__ )))

from feature_store .store import FeatureStore ,LRUCache
from feature_store .rolling_engine import FeatureSnapshot


class TestLRUCache :

    def test_basic_set_get (self ):
        cache =LRUCache (capacity =3 )
        snap =FeatureSnapshot (entity_id ="a",computed_at =0.0 )
        cache .set ("a",snap )
        result =cache .get ("a")
        assert result is not None
        assert result .entity_id =="a"

    def test_miss_returns_none (self ):
        cache =LRUCache (capacity =3 )
        assert cache .get ("nonexistent")is None

    def test_eviction_on_overflow (self ):
        cache =LRUCache (capacity =3 )
        for i in range (4 ):
            cache .set (str (i ),FeatureSnapshot (entity_id =str (i ),computed_at =0.0 ))

        assert cache .get ("0")is None

        assert cache .get ("3")is not None

    def test_access_updates_lru_order (self ):
        cache =LRUCache (capacity =3 )
        for i in range (3 ):
            cache .set (str (i ),FeatureSnapshot (entity_id =str (i ),computed_at =0.0 ))


        cache .get ("0")

        cache .set ("3",FeatureSnapshot (entity_id ="3",computed_at =0.0 ))

        assert cache .get ("0")is not None ,"'0' was recently accessed, must survive"
        assert cache .get ("1")is None ,"'1' is LRU, must be evicted"

    def test_hit_rate_calculation (self ):
        cache =LRUCache (capacity =10 )
        snap =FeatureSnapshot (entity_id ="x",computed_at =0.0 )
        cache .set ("x",snap )

        cache .get ("x")
        cache .get ("x")
        cache .get ("y")

        rate =cache .hit_rate ()
        assert abs (rate -2 /3 )<1e-9

    def test_thread_safe_concurrent_access (self ):
        cache =LRUCache (capacity =100 )
        errors =[]

        def _worker (tid :int )->None :
            try :
                for i in range (500 ):
                    key =f"key_{(tid *100 +i )%50 }"
                    snap =FeatureSnapshot (entity_id =key ,computed_at =float (i ))
                    cache .set (key ,snap )
                    cache .get (key )
            except Exception as exc :
                errors .append (exc )

        threads =[threading .Thread (target =_worker ,args =(i ,))for i in range (8 )]
        for t in threads :t .start ()
        for t in threads :t .join ()

        assert not errors


class TestFeatureStore :

    @pytest .fixture
    def store (self ):
        """Return a store with Redis disabled (unit-test safe)."""
        s =FeatureStore (
        redis_host ="127.0.0.1",
        redis_port =6380 ,
        l1_capacity =100 ,
        buffer_capacity =200 ,
        )
        return s

    def test_ingest_and_get (self ,store ):
        store .ingest ("alice",100.0 ,lat =40.7 ,lon =-74.0 ,merchant ="test_shop")
        snap =store .get_features ("alice",current_amount =100.0 )
        assert snap .entity_id =="alice"
        assert snap .txn_count_60s >=1

    def test_l1_cache_hit_after_ingest (self ,store ):
        store .ingest ("bob",50.0 )

        hit_before =store ._l1 .hit_rate ()
        store .get_features ("bob")
        hit_after =store ._l1 .hit_rate ()

        assert store ._l1 ._hits >0

    def test_multiple_ingests_accumulate (self ,store ):
        for amt in [10.0 ,20.0 ,30.0 ]:
            store .ingest ("carol",amt )
        snap =store .get_features ("carol")
        assert snap .txn_count_60s ==3
        assert abs (snap .amount_sum_60s -60.0 )<1e-6

    def test_unknown_entity_returns_empty_snap (self ,store ):
        snap =store .get_features ("nobody_0000",current_amount =42.0 )
        assert snap .txn_count_60s ==0
        assert snap .current_amount ==42.0

    def test_health_report (self ,store ):
        store .ingest ("dave",77.0 )
        h =store .health ()
        assert h ["writes"]>=1
        assert h ["tracked_entities"]>=1
        assert "l1_cache_hit_rate"in h
        assert "l2_available"in h

    def test_write_count_increments (self ,store ):
        n =5
        for i in range (n ):
            store .ingest (f"entity_{i }",10.0 )
        assert store ._write_count ==n

    def test_feature_max_age_triggers_recompute (self ,store ):
        store .ingest ("eve",100.0 )

        snap =store ._l1 .get ("eve")
        if snap :
            snap .computed_at =time .time ()-100
            store ._l1 .set ("eve",snap )


        result =store .get_features ("eve",max_age_seconds =5 )
        assert (time .time ()-result .computed_at )<5

    def test_concurrent_ingest (self ,store ):
        """Many threads writing to different entities simultaneously."""
        errors =[]

        def _ingest (entity_id :str )->None :
            try :
                for _ in range (50 ):
                    store .ingest (entity_id ,25.0 )
            except Exception as exc :
                errors .append (exc )

        threads =[threading .Thread (target =_ingest ,args =(f"concurrent_{i }",))
        for i in range (10 )]
        for t in threads :t .start ()
        for t in threads :t .join ()

        assert not errors

        assert store ._l1 .size ()>0

    def test_feature_snapshot_model_array_stable (self ,store ):
        """to_model_array() must always return the same shape with no NaN/Inf."""
        import numpy as np
        for amt in [1 ,100 ,10_000 ]:
            store .ingest ("fred",float (amt ))
        snap =store .get_features ("fred",current_amount =5000.0 )
        arr =snap .to_model_array ()
        assert arr .shape ==(20 ,)
        assert not np .any (np .isnan (arr ))
        assert not np .any (np .isinf (arr ))


@pytest .mark .redis
class TestRedisIntegration :
    """
    These tests require a running Redis on localhost:6379.
    Mark with -m redis to run selectively:
        pytest tests/ -m redis
    """

    @pytest .fixture
    def redis_store (self ):
        s =FeatureStore (redis_host ="localhost",redis_port =6379 ,l1_capacity =10 )
        if not s ._l2 .available :
            pytest .skip ("Redis not available")
        return s

    def test_l2_write_read (self ,redis_store ):
        redis_store .ingest ("redis_user",200.0 )

        redis_store ._l1 ._store .clear ()
        snap =redis_store .get_features ("redis_user")
        assert snap .txn_count_60s >=1

    def test_l2_populates_l1 (self ,redis_store ):
        redis_store .ingest ("redis_user2",300.0 )
        redis_store ._l1 ._store .clear ()

        assert redis_store ._l1 .get ("redis_user2")is None
        redis_store .get_features ("redis_user2")
        assert redis_store ._l1 .get ("redis_user2")is not None

    def test_pipeline_batch_write (self ,redis_store ):
        snaps =[
        FeatureSnapshot (entity_id =f"batch_{i }",computed_at =time .time ())
        for i in range (5 )
        ]
        redis_store ._l2 .pipeline_set (snaps )
        for snap in snaps :
            result =redis_store ._l2 .get (snap .entity_id )
            assert result is not None
