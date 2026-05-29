from __future__ import annotations

import os
import sys
import time
import pickle
import tempfile

import numpy as np
import pytest

sys .path .insert (0 ,os .path .dirname (os .path .dirname (__file__ )))

from feature_store .store import FeatureStore
from feature_store .rolling_engine import FeatureSnapshot


def _make_dummy_model_pkl (path :str )->None :
    """Create a minimal pickled model bundle with a scikit-learn classifier."""
    from sklearn .linear_model import LogisticRegression
    import numpy as np

    X =np .random .rand (100 ,20 )
    y =(np .random .rand (100 )>0.9 ).astype (int )
    clf =LogisticRegression (max_iter =10 ).fit (X ,y )

    payload ={
    "model":clf ,
    "scaler":None ,
    "feature_names":[f"f{i }"for i in range (20 )],
    "model_type":"logistic_regression",
    "roc_auc":0.75 ,
    "pr_auc":0.3 ,
    "threshold":0.5 ,
    "version":"test-1.0",
    }
    with open (path ,"wb")as f :
        pickle .dump (payload ,f )


@pytest .fixture (scope ="module")
def tmp_model (tmp_path_factory ):
    d =tmp_path_factory .mktemp ("models")
    p =str (d /"fraud_model.pkl")
    _make_dummy_model_pkl (p )
    return p


@pytest .fixture
def store ():
    return FeatureStore (redis_port =6380 ,l1_capacity =100 )


class TestFraudModel :

    def test_load_dummy_model (self ,tmp_model ):
        from inference_engine .server import FraudModel
        m =FraudModel (tmp_model )
        assert m .version =="test-1.0"

    def test_predict_returns_triple (self ,tmp_model ):
        from inference_engine .server import FraudModel
        m =FraudModel (tmp_model )
        arr =np .zeros (20 ,dtype =np .float32 )
        prob ,decision ,confidence =m .predict (arr )
        assert 0.0 <=prob <=1.0
        assert decision in ("APPROVE","REVIEW","DECLINE")
        assert 0.0 <=confidence <=1.0

    def test_decision_thresholds (self ,tmp_model ):
        from inference_engine .server import FraudModel
        m =FraudModel (tmp_model )

        class MockProba :
            def predict_proba (self ,X ):
                return np .array ([[1 -self ._p ,self ._p ]])
            _p =0.0

        m ._model =MockProba ()


        m ._model ._p =0.1
        _ ,d ,_ =m .predict (np .zeros (20 ,dtype =np .float32 ))
        assert d =="APPROVE"


        m ._model ._p =0.45
        _ ,d ,_ =m .predict (np .zeros (20 ,dtype =np .float32 ))
        assert d =="REVIEW"


        m ._model ._p =0.85
        _ ,d ,_ =m .predict (np .zeros (20 ,dtype =np .float32 ))
        assert d =="DECLINE"

    def test_reload (self ,tmp_model ):
        from inference_engine .server import FraudModel
        m =FraudModel (tmp_model )

        m .reload ()
        assert m .version =="test-1.0"


class TestInferenceServiceCore :

    def test_single_predict_structure (self ,tmp_model ,store ):
        from inference_engine .server import FraudModel ,InferenceServiceCore
        model =FraudModel (tmp_model )
        core =InferenceServiceCore (store ,model )

        result =core .predict_single (
        entity_id ="test_user",
        transaction_id ="txn_001",
        amount =150.0 ,
        lat =40.7 ,
        lon =-74.0 ,
        merchant_category ="grocery",
        )
        assert "fraud_probability"in result
        assert "decision"in result
        assert "inference_latency_us"in result
        assert "feature_latency_us"in result
        assert 0.0 <=result ["fraud_probability"]<=1.0
        assert result ["decision"]in ("APPROVE","REVIEW","DECLINE")

    def test_latency_is_positive (self ,tmp_model ,store ):
        from inference_engine .server import FraudModel ,InferenceServiceCore
        core =InferenceServiceCore (store ,FraudModel (tmp_model ))
        r =core .predict_single ("u","t",10.0 )
        assert r ["inference_latency_us"]>0
        assert r ["feature_latency_us"]>=0

    def test_request_counter_increments (self ,tmp_model ,store ):
        from inference_engine .server import FraudModel ,InferenceServiceCore
        core =InferenceServiceCore (store ,FraudModel (tmp_model ))
        assert core .request_count ==0
        core .predict_single ("u1","t1",50.0 )
        core .predict_single ("u2","t2",75.0 )
        assert core .request_count ==2

    def test_include_features_flag (self ,tmp_model ,store ):
        from inference_engine .server import FraudModel ,InferenceServiceCore
        core =InferenceServiceCore (store ,FraudModel (tmp_model ))

        r_no_feats =core .predict_single ("u","t",50.0 ,include_features =False )
        r_feats =core .predict_single ("u","t",50.0 ,include_features =True )

        assert r_no_feats ["features"]is None
        assert r_feats ["features"]is not None
        assert "txn_count_60s"in r_feats ["features"]

    def test_batch_predict_length (self ,tmp_model ,store ):
        from inference_engine .server import FraudModel ,InferenceServiceCore
        core =InferenceServiceCore (store ,FraudModel (tmp_model ))
        requests =[
        dict (entity_id =f"u{i }",transaction_id =f"t{i }",amount =float (i *10 ))
        for i in range (5 )
        ]
        results =core .batch_predict (requests )
        assert len (results )==5
        assert all (r ["decision"]in ("APPROVE","REVIEW","DECLINE")for r in results )

    def test_history_improves_feature_richness (self ,tmp_model ,store ):
        """After seeding history, feature count should be non-zero."""
        from inference_engine .server import FraudModel ,InferenceServiceCore
        core =InferenceServiceCore (store ,FraudModel (tmp_model ))


        for amt in [50 ,60 ,70 ,80 ]:
            store .ingest ("rich_user",float (amt ))

        r =core .predict_single ("rich_user","t",65.0 ,include_features =True )
        feats =r ["features"]
        assert feats ["txn_count_60s"]>=4

    def test_model_version_in_response (self ,tmp_model ,store ):
        from inference_engine .server import FraudModel ,InferenceServiceCore
        core =InferenceServiceCore (store ,FraudModel (tmp_model ))
        r =core .predict_single ("u","t",99.0 )
        assert r ["model_version"]=="test-1.0"
