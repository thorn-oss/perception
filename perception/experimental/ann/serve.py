# pylint: disable=line-too-long,too-many-arguments
import json
import typing
import asyncio
import logging
import functools

import aiohttp.web
import numpy as np
import pandas as pd
from pythonjsonlogger import jsonlogger

import perception.hashers.tools as pht
from .index import ApproximateNearestNeighbors


def is_similarity_valid(data, index: ApproximateNearestNeighbors):
    """Validates input to the similarity endpoint."""
    hash_format = data.get('hash_format', 'base64')
    expected_string_length = pht.get_string_length(
        hash_length=index.hash_length,
        dtype=index.dtype,
        hash_format=hash_format)
    return (isinstance(data, dict) and "queries" in data
            and isinstance(data["queries"], list) and all(
                isinstance(x.get('hash', None), str) for x in data["queries"])
            and hash_format in ['hex', 'base64'] and all(
                len(x.get('hash', None)) == expected_string_length
                for x in data['queries']))


async def similarity(request):
    """Responds to a vector similarity query of the form:

    ```
    {
        "queries": [{"id": str, "hash": "base64_encoded_hash1"}, ...],
        "k": int,
        "threshold": float,
        "hash_format": "base64"
    }
    ```

    with information about similar vectors in the index in the form:

    ```
    {
      "queries": [{"id": str, "matches": [{"metadata": {json metadata}, "distance": float},...],...]
    }
    ```
    """
    try:
        request_data = await request.json()
    except json.JSONDecodeError:
        return aiohttp.web.json_response({
            "reason": "Malformed JSON"
        },
                                         status=400)

    index = request.app['index']
    try:
        assert is_similarity_valid(request_data, index)
    except:  # pylint: disable=bare-except
        return aiohttp.web.json_response({
            "reason": "Invalid JSON request"
        },
                                         status=400)

    async with request.app["query_semaphore"]:
        matches = await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(
                index.search,
                queries=request_data["queries"],
                threshold=request_data.get("threshold",
                                           request.app["default_threshold"]),
                threshold_func=request.app["default_threshold_func"],
                k=request_data.get("k", request.app["default_k"]),
                hash_format=request_data.get("hash_format", "base64")))
        matches = json.loads(pd.io.json.dumps({'queries': matches}))

    return aiohttp.web.json_response(matches)


def get_logger(name, log_level):
    logger = logging.Logger(name=name, level=log_level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s:%(levelname)s:%(name)s:%(message)s%(exc_info)"))
    logger.addHandler(handler)
    return logger


async def serve(
        index: ApproximateNearestNeighbors,
        default_threshold: int = None,
        default_threshold_func: typing.Callable[[np.ndarray], float] = None,
        default_k: int = 1,
        concurrency: int = 2,
        log_level=logging.INFO,
        host='localhost',
        port=8080):
    """Serve an index as a web API. This function does not block.
    If you wish to use the function in a blocking manner, you can
    do something like

    .. code-block:: python

        loop = asyncio.get_event_loop()
        loop.run_until_complete(serve(...))
        loop.run_forever()

    You can query the API with something like:

    .. code-block:: bash

        curl --header "Content-Type: application/json" \\
             --request POST \\
             --data '{"queries": [{"hash": "<hash string>", "id": "bar"}], "threshold": 1200}' \\
             http://localhost:8080/v1/similarity

    Args:
        index: The underlying index
        default_threshold: The default threshold for matches
        default_k: The default number of nearest neighbors to look for
        concurrency: The number of concurrent requests served
        log_level: The log level to use for the logger
        host: The host for the servoce
        port: The port for the service
    """
    logger = get_logger(name='serve', log_level=log_level)
    logger.info("Initializing web service")
    app = aiohttp.web.Application()
    app.router.add_post('/v1/similarity', similarity, name='similarity')

    # Store globals in the application object
    app["default_threshold"] = default_threshold
    app["logger"] = logger
    app["default_k"] = default_k
    app["default_threshold_func"] = default_threshold_func
    app["index"] = index
    app["query_semaphore"] = asyncio.Semaphore(concurrency)
    logger.info("Entering web service listener loop.")
    runner = aiohttp.web.AppRunner(app, logger=logger)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, host, port)
    await site.start()
    return site
