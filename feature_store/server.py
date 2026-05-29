from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from typing import Optional

from aiohttp import web

from .store import FeatureStore

sys .path .insert (0 ,os .path .dirname (os .path .dirname (__file__ )))

logger =logging .getLogger (__name__ )


class HealthServer :
    """Tiny HTTP server for /health, /stats, /metrics endpoints."""

    def __init__ (self ,store :FeatureStore ,port :int =8080 ):
        self ._store =store
        self ._port =port
        self ._app =web .Application ()
        self ._app .router .add_get ("/health",self ._health )
        self ._app .router .add_get ("/stats",self ._stats )
        self ._app .router .add_get ("/ready",self ._ready )
        self ._runner :Optional [web .AppRunner ]=None

    async def start (self )->None :
        self ._runner =web .AppRunner (self ._app )
        await self ._runner .setup ()
        site =web .TCPSite (self ._runner ,"0.0.0.0",self ._port )
        await site .start ()
        logger .info ("Health server listening on http://0.0.0.0:%d",self ._port )

    async def stop (self )->None :
        if self ._runner :
            await self ._runner .cleanup ()

    async def _health (self ,_ :web .Request )->web .Response :
        return web .json_response ({"status":"ok","timestamp":time .time ()})

    async def _ready (self ,_ :web .Request )->web .Response :
        h =self ._store .health ()

        ready =h .get ("writes",0 )>0
        status =200 if ready else 503
        return web .json_response ({"ready":ready ,**h },status =status )

    async def _stats (self ,_ :web .Request )->web .Response :
        return web .json_response (self ._store .health ())


async def kafka_ingest_loop (store :FeatureStore ,bootstrap :str )->None :
    """Consume transactions from Kafka and feed them into the FeatureStore."""
    from stream_ingestion .simulator import TransactionConsumer

    consumer =TransactionConsumer (bootstrap_servers =bootstrap )
    await consumer .start ()
    logger .info ("Kafka consumer started, ingesting into FeatureStore …")

    ingested =0
    t_report =time .time ()

    async for txn in consumer :
        store .ingest (
        entity_id =txn .entity_id ,
        amount =txn .amount ,
        lat =txn .latitude ,
        lon =txn .longitude ,
        merchant =txn .merchant_id ,
        merchant_category =txn .merchant_category ,
        ts =txn .timestamp_ms /1000.0 ,
        )
        ingested +=1

        if time .time ()-t_report >=5.0 :
            rps =ingested /max (time .time ()-t_report ,0.001 )
            logger .info ("Ingested %d events | Store writes: %d | ~%.0f RPS",
            ingested ,store .health ()["writes"],rps )
            ingested =0
            t_report =time .time ()


async def simulator_ingest_loop (store :FeatureStore ,target_rps :int )->None :
    """Run the built-in simulator and feed events directly into the FeatureStore."""
    from stream_ingestion .simulator import TransactionSimulator

    def _on_txn (txn )->None :
        store .ingest (
        entity_id =txn .entity_id ,
        amount =txn .amount ,
        lat =txn .latitude ,
        lon =txn .longitude ,
        merchant =txn .merchant_id ,
        merchant_category =txn .merchant_category ,
        )

    sim =TransactionSimulator (
    n_normal =400 ,
    n_churner =50 ,
    n_fraudster =50 ,
    target_rps =target_rps ,
    on_transaction =_on_txn ,
    )

    logger .info ("Simulator starting — target %d RPS …",target_rps )

    await sim .run (duration_seconds =0 )


async def _main (args :argparse .Namespace )->None :
    logging .basicConfig (
    level =logging .INFO ,
    format ="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt ="%H:%M:%S",
    )

    store =FeatureStore (
    redis_host =args .redis_host ,
    redis_port =args .redis_port ,
    l1_capacity =args .l1_capacity ,
    buffer_capacity =args .buffer_capacity ,
    max_entities =args .max_entities ,
    )
    logger .info ("FeatureStore initialised (L2=%s)","Redis"if store ._l2 .available else "disabled")

    health_server =HealthServer (store ,port =args .http_port )


    loop =asyncio .get_event_loop ()
    stop_event =asyncio .Event ()

    def _shutdown (*_ ):
        logger .info ("Shutdown signal received.")
        stop_event .set ()

    for sig in (signal .SIGINT ,signal .SIGTERM ):
        try :
            loop .add_signal_handler (sig ,_shutdown )
        except NotImplementedError :

            pass


    await health_server .start ()


    if args .simulate :
        ingest_task =asyncio .create_task (
        simulator_ingest_loop (store ,target_rps =args .rps )
        )
    else :
        ingest_task =asyncio .create_task (
        kafka_ingest_loop (store ,bootstrap =args .kafka )
        )


    await stop_event .wait ()
    ingest_task .cancel ()
    try :
        await ingest_task
    except asyncio .CancelledError :
        pass

    await health_server .stop ()
    logger .info ("Feature store server stopped. Final stats: %s",store .health ())


def main ()->None :
    parser =argparse .ArgumentParser (description ="Feature Store Server")
    parser .add_argument ("--simulate",action ="store_true",help ="Use built-in simulator instead of Kafka")
    parser .add_argument ("--rps",type =int ,default =1000 ,help ="Target RPS for simulator")
    parser .add_argument ("--kafka",type =str ,default ="localhost:9092")
    parser .add_argument ("--redis-host",type =str ,default ="localhost")
    parser .add_argument ("--redis-port",type =int ,default =6379 )
    parser .add_argument ("--http-port",type =int ,default =8080 )
    parser .add_argument ("--l1-capacity",type =int ,default =10_000 )
    parser .add_argument ("--buffer-capacity",type =int ,default =1_000 )
    parser .add_argument ("--max-entities",type =int ,default =100_000 )
    args =parser .parse_args ()

    asyncio .run (_main (args ))


if __name__ =="__main__":
    main ()
