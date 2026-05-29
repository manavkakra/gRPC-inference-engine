from __future__ import annotations

import math
import threading
import time
from collections import defaultdict ,deque
from dataclasses import dataclass ,field
from typing import Dict ,List ,Optional ,Tuple

import numpy as np


@dataclass
class WindowStats :
    """Statistics computed over one sliding window."""
    window_seconds :int
    count :int =0
    total :float =0.0
    mean :float =0.0
    std :float =0.0
    min_v :float =0.0
    max_v :float =0.0


@dataclass
class FeatureSnapshot :
    """Complete feature vector for one entity at one point in time."""
    entity_id :str
    computed_at :float


    amount_sum_1s :float =0.0
    amount_mean_1s :float =0.0
    amount_std_1s :float =0.0
    txn_count_1s :int =0


    amount_sum_5s :float =0.0
    amount_mean_5s :float =0.0
    amount_std_5s :float =0.0
    txn_count_5s :int =0
    amount_max_5s :float =0.0


    amount_sum_60s :float =0.0
    amount_mean_60s :float =0.0
    amount_std_60s :float =0.0
    txn_count_60s :int =0
    amount_max_60s :float =0.0
    amount_min_60s :float =0.0


    amount_zscore :float =0.0
    velocity_change_ratio :float =0.0
    unique_merchants_60s :int =0
    geo_distance_delta :float =0.0
    current_amount :float =0.0
    merchant_category :str =""

    def to_dict (self )->dict :
        return {k :v for k ,v in self .__dict__ .items ()}

    def to_model_array (self )->np .ndarray :
        """Return a fixed-length float array for ML model input."""
        return np .array ([
        self .amount_sum_1s ,self .amount_mean_1s ,self .amount_std_1s ,self .txn_count_1s ,
        self .amount_sum_5s ,self .amount_mean_5s ,self .amount_std_5s ,self .txn_count_5s ,self .amount_max_5s ,
        self .amount_sum_60s ,self .amount_mean_60s ,self .amount_std_60s ,self .txn_count_60s ,
        self .amount_max_60s ,self .amount_min_60s ,
        self .amount_zscore ,self .velocity_change_ratio ,
        self .unique_merchants_60s ,self .geo_distance_delta ,self .current_amount ,
        ],dtype =np .float32 )


class EntityRingBuffer :
    """
    Fixed-capacity ring buffer storing per-transaction data for one entity.

    Layout (all NumPy for cache locality):
        timestamps  – float64[N]  Unix seconds
        amounts     – float64[N]
        latitudes   – float64[N]
        longitudes  – float64[N]

    Merchants stored in a parallel Python deque (strings are not NumPy-native).
    """

    __slots__ =(
    "_cap","_head","_size",
    "timestamps","amounts","latitudes","longitudes",
    "merchants","_lock",
    "last_lat","last_lon",
    )

    def __init__ (self ,capacity :int =1000 ):
        self ._cap =capacity
        self ._head =0
        self ._size =0

        self .timestamps =np .zeros (capacity ,dtype =np .float64 )
        self .amounts =np .zeros (capacity ,dtype =np .float64 )
        self .latitudes =np .zeros (capacity ,dtype =np .float64 )
        self .longitudes =np .zeros (capacity ,dtype =np .float64 )
        self .merchants =deque (maxlen =capacity )

        self ._lock =threading .RLock ()
        self .last_lat =0.0
        self .last_lon =0.0

    def push (
    self ,
    ts :float ,
    amount :float ,
    lat :float ,
    lon :float ,
    merchant :str ,
    )->None :
        with self ._lock :
            self .last_lat =lat
            self .last_lon =lon

            idx =self ._head
            self .timestamps [idx ]=ts
            self .amounts [idx ]=amount
            self .latitudes [idx ]=lat
            self .longitudes [idx ]=lon


            if len (self .merchants )==self ._cap :
                self .merchants .popleft ()
            self .merchants .append (merchant )

            self ._head =(self ._head +1 )%self ._cap
            if self ._size <self ._cap :
                self ._size +=1

    def _window_slice (self ,since_ts :float )->Tuple [np .ndarray ,np .ndarray ,List [str ]]:
        """
        Return (amounts, timestamps, merchants) for events after since_ts.
        Handles wrap-around correctly.
        """
        if self ._size ==0 :
            return np .empty (0 ),np .empty (0 ),[]


        if self ._size <self ._cap :
            ts_view =self .timestamps [:self ._size ]
            amt_view =self .amounts [:self ._size ]
        else :

            tail =self ._head
            ts_view =np .concatenate ([self .timestamps [tail :],self .timestamps [:tail ]])
            amt_view =np .concatenate ([self .amounts [tail :],self .amounts [:tail ]])

        mask =ts_view >=since_ts
        filtered_ts =ts_view [mask ]
        filtered_amt =amt_view [mask ]


        merchant_list =list (self .merchants )
        n_keep =int (mask .sum ())
        filtered_merchants =merchant_list [-n_keep :]if n_keep >0 else []

        return filtered_amt ,filtered_ts ,filtered_merchants

    def window_stats (self ,window_seconds :int )->WindowStats :
        since =time .time ()-window_seconds
        with self ._lock :
            amounts ,_ ,_ =self ._window_slice (since )

        stats =WindowStats (window_seconds =window_seconds )
        if len (amounts )==0 :
            return stats

        stats .count =len (amounts )
        stats .total =float (np .sum (amounts ))
        stats .mean =float (np .mean (amounts ))
        stats .std =float (np .std (amounts ))if len (amounts )>1 else 0.0
        stats .min_v =float (np .min (amounts ))
        stats .max_v =float (np .max (amounts ))
        return stats

    def unique_merchants (self ,window_seconds :int )->int :
        since =time .time ()-window_seconds
        with self ._lock :
            _ ,_ ,merchants =self ._window_slice (since )
        return len (set (merchants ))

    def last_location (self )->Tuple [float ,float ]:
        with self ._lock :
            return self .last_lat ,self .last_lon


_R_KM =6371.0

def haversine_km (lat1 :float ,lon1 :float ,lat2 :float ,lon2 :float )->float :
    """Great-circle distance in km between two lat/lon points."""
    if lat1 ==0.0 and lon1 ==0.0 :
        return 0.0
    phi1 ,phi2 =math .radians (lat1 ),math .radians (lat2 )
    dphi =math .radians (lat2 -lat1 )
    dlam =math .radians (lon2 -lon1 )
    a =math .sin (dphi /2 )**2 +math .cos (phi1 )*math .cos (phi2 )*math .sin (dlam /2 )**2
    return 2.0 *_R_KM *math .asin (math .sqrt (a ))


class RollingFeatureEngine :
    """
    Manages per-entity ring buffers and computes feature snapshots.

    Thread safety:
      • EntityRingBuffer.push() is lock-protected per entity.
      • compute_features() reads are lock-protected per buffer.
      • The entity registry (_buffers) itself is protected by _registry_lock.
    """

    WINDOWS =[1 ,5 ,60 ]
    MERCHANT_CATEGORY_MAP ={
    "grocery":0 ,"gas_station":1 ,"restaurant":2 ,"online":3 ,
    "atm":4 ,"travel":5 ,"retail":6 ,"other":7 ,
    }

    def __init__ (
    self ,
    buffer_capacity :int =1000 ,
    max_entities :int =100_000 ,
    ):
        self ._cap =buffer_capacity
        self ._max_entities =max_entities
        self ._buffers :Dict [str ,EntityRingBuffer ]={}
        self ._registry_lock =threading .RLock ()


        self ._total_events =0
        self ._total_computes =0


    def ingest (
    self ,
    entity_id :str ,
    amount :float ,
    lat :float ,
    lon :float ,
    merchant :str ,
    ts :Optional [float ]=None ,
    )->None :
        """Record a new transaction for an entity (non-blocking hot path)."""
        ts =ts or time .time ()
        buf =self ._get_or_create (entity_id )
        prev_lat ,prev_lon =buf .last_location ()
        buf .push (ts ,amount ,lat ,lon ,merchant )
        self ._total_events +=1

    def _get_or_create (self ,entity_id :str )->EntityRingBuffer :

        buf =self ._buffers .get (entity_id )
        if buf is not None :
            return buf

        with self ._registry_lock :
            buf =self ._buffers .get (entity_id )
            if buf is None :
                if len (self ._buffers )>=self ._max_entities :

                    evict_key =next (iter (self ._buffers ))
                    del self ._buffers [evict_key ]
                buf =EntityRingBuffer (capacity =self ._cap )
                self ._buffers [entity_id ]=buf
            return buf


    def compute_features (
    self ,
    entity_id :str ,
    current_amount :float =0.0 ,
    current_lat :float =0.0 ,
    current_lon :float =0.0 ,
    merchant_category :str ="other",
    )->FeatureSnapshot :
        """
        Compute a full FeatureSnapshot for entity_id.
        Returns a zero-filled snapshot if entity has no history.
        """
        snap =FeatureSnapshot (
        entity_id =entity_id ,
        computed_at =time .time (),
        current_amount =current_amount ,
        merchant_category =merchant_category ,
        )

        buf =self ._buffers .get (entity_id )
        if buf is None :
            return snap


        s1 =buf .window_stats (1 )
        snap .amount_sum_1s =s1 .total
        snap .amount_mean_1s =s1 .mean
        snap .amount_std_1s =s1 .std
        snap .txn_count_1s =s1 .count


        s5 =buf .window_stats (5 )
        snap .amount_sum_5s =s5 .total
        snap .amount_mean_5s =s5 .mean
        snap .amount_std_5s =s5 .std
        snap .txn_count_5s =s5 .count
        snap .amount_max_5s =s5 .max_v


        s60 =buf .window_stats (60 )
        snap .amount_sum_60s =s60 .total
        snap .amount_mean_60s =s60 .mean
        snap .amount_std_60s =s60 .std
        snap .txn_count_60s =s60 .count
        snap .amount_max_60s =s60 .max_v
        snap .amount_min_60s =s60 .min_v


        if s60 .std >0 :
            snap .amount_zscore =(current_amount -s60 .mean )/s60 .std
        else :
            snap .amount_zscore =0.0


        snap .velocity_change_ratio =(
        (snap .txn_count_1s /max (snap .txn_count_60s ,1 ))*60.0
        )

        snap .unique_merchants_60s =buf .unique_merchants (60 )


        prev_lat ,prev_lon =buf .last_location ()
        snap .geo_distance_delta =haversine_km (prev_lat ,prev_lon ,current_lat ,current_lon )

        self ._total_computes +=1
        return snap


    def stats (self )->dict :
        return {
        "tracked_entities":len (self ._buffers ),
        "total_events_ingested":self ._total_events ,
        "total_features_computed":self ._total_computes ,
        }
