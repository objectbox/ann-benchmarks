from os import path
import numpy as np
from typing import *
from multiprocessing.pool import ThreadPool
import threading
from time import time
from ..base.module import BaseANN
import objectbox
from objectbox import *
from objectbox.model import *
from objectbox.model.properties import HnswIndex, VectorDistanceType
from objectbox.c import *


def _convert_metric_to_distance_type(metric: str) -> VectorDistanceType:
    if metric == 'euclidean':
        return VectorDistanceType.EUCLIDEAN
    elif metric == 'angular':
        return VectorDistanceType.COSINE
    else:
        raise ValueError(f"Metric type not supported: {metric}")


def _create_entity_class(dimensions: int, distance_type: VectorDistanceType, m: int, ef_construction: int):
    """ Dynamically define an Entity class according to the parameters. """

    @Entity(id=1, uid=1)
    class VectorEntity:
        id = Id(id=1, uid=1001)
        vector = Property(np.ndarray, type=PropertyType.floatVector, id=2, uid=1002,
                          index=HnswIndex(
                              id=1, uid=10001,
                              dimensions=dimensions,
                              distance_type=distance_type,
                              neighbors_per_node=m,
                              indexing_search_count=ef_construction
                          ))

    return VectorEntity


class ObjectBox(BaseANN):
    _db_path: str
    _store: ObjectBox

    def __init__(self, metric, dimensions, m: int, ef_construction: int):
        print(f"[objectbox] Version: {objectbox.version}")
        print(f"[objectbox] Metric: {metric}, "
              f"Dimensions: {dimensions}, "
              f"M: {m}, "
              f"ef construction: {ef_construction}")

        self._dimensions = dimensions
        self._distance_type = _convert_metric_to_distance_type(metric)
        self._m = m
        self._ef_construction = ef_construction

        self._db_path = "./objectbox-benchmark-db"
        objectbox.Store.remove_db_files(self._db_path)  # Remove any preexisting DB; e.g. let IDs start at 1

        print(f"[objectbox] DB path: \"{self._db_path}\"")

        self._entity_class = _create_entity_class(
            self._dimensions, self._distance_type, self._m, self._ef_construction)

        model = objectbox.Model()
        model.entity(self._entity_class, last_property_id=IdUid(2, 1002))
        model.last_entity_id = IdUid(1, 1)
        model.last_index_id = IdUid(1, 10001)

        self._store = objectbox.Store(
                model=model,
                directory=self._db_path,
                max_db_size_in_kb=2097152
        )
        self._vector_property = self._entity_class.get_property("vector")

        self._box = self._store.box(self._entity_class)

        self._batch_results = None
        self._ef_search = None
        self._thread_local = threading.local()

    def done(self):
        print(f"[objectbox] Done!")

    def fit(self, x: np.array) -> None:
        BATCH_SIZE = 10000

        num_objects, vector_dim = x.shape
        num_inserted_objects = self._box.count()
        started_at = time()

        print(f"[objectbox] Already inserted objects: {num_inserted_objects} ({num_objects})")

        if num_inserted_objects != num_objects:
            self._box.remove_all()
            for i in range(0, num_objects, BATCH_SIZE):
                self._box.put(*[self._entity_class(vector=vector) for vector in x[i:i + BATCH_SIZE]])
                print(f"[objectbox] Inserted {i + BATCH_SIZE}/{num_objects} objects... "
                      f"Elapsed time: {time() - started_at:.1f}s")
            assert self._box.count() == num_objects
        else:
            print(f"[objectbox] DB was already filled!")

    def set_query_arguments(self, ef: float):
        print(f"[objectbox] Query search params; EF: {ef}")
        self._ef_search = ef
        self._thread_local = threading.local()  # reset as all threads need to rebuild the query for the new efSearch

    def query(self, q: np.array, n: int) -> np.array:
        if not hasattr(self._thread_local,'query'):
            self._thread_local.query = self._box.query(self._vector_property.nearest_neighbor(q, self._ef_search).alias("q")).build()
        query = self._thread_local.query
        query.limit(n)
        query.set_parameter_alias_vector_f32("q", q)
        return query.find_ids_by_score_numpy() - 1  # Because OBX IDs start at 1

    def batch_query(self, q_batch: np.array, n: int) -> None:
        print(f"[objectbox] Query batch shape: {q_batch.shape}; N: {n}")

        def _run_batch_query(q: np.ndarray):
            return self.query(q, n)

        pool = ThreadPool()
        self._batch_results = pool.map(lambda q: _run_batch_query(q), q_batch)

    def get_batch_results(self) -> np.array:
        return self._batch_results

    def __str__(self) -> str:
        return f"objectbox(" \
               f"dimensions={self._dimensions}, " \
               f"m={self._m}, " \
               f"ef_construction={self._ef_construction}, " \
               f"ef_search={self._ef_search}" \
               f")"
