from os import path
import numpy as np
from typing import *
from multiprocessing.pool import ThreadPool
import threading
from ..base.module import BaseANN
import objectbox
from objectbox import *
from objectbox.model import *
from objectbox.model.properties import HnswIndex, HnswDistanceType
from objectbox.c import *


def create_entity_class(dimensions: int, m: int, ef_construction: int):
    """ Dynamically define an Entity class according to the parameters. """

    @Entity(id=1, uid=1)
    class VectorEntity:
        id = Id(id=1, uid=1001)
        vector = Property(np.ndarray, type=PropertyType.floatVector, id=2, uid=1002,
                          index=HnswIndex(
                              id=1, uid=10001,
                              dimensions=dimensions,
                              distance_type=HnswDistanceType.EUCLIDEAN,
                              neighbors_per_node=m,
                              indexing_search_count=ef_construction
                          ))

    return VectorEntity


class ObjectBox(BaseANN):
    _db_path: str
    _ob: ObjectBox

    def __init__(self, metric, dimensions, m: int, ef_construction: int):
        print(f"[objectbox] Version: {objectbox.version}")
        print(f"[objectbox] Metric: {metric}, Dimensions: {dimensions}, M: {m}, ef construction: {ef_construction}")

        self._dimensions = dimensions
        self._m = m
        self._ef_construction = ef_construction

        self._db_path = "./test-db"
        print(f"[objectbox] DB path: \"{self._db_path}\"")

        self._entity_class = create_entity_class(dimensions, m, ef_construction)

        model = objectbox.Model()
        model.entity(self._entity_class, last_property_id=IdUid(2, 1002))
        model.last_entity_id = IdUid(1, 1)
        model.last_index_id = IdUid(1, 10001)

        self._ob = objectbox.Builder().model(model).directory(self._db_path).build()
        self._vector_property = self._entity_class.get_property("vector")

        self._box = objectbox.Box(self._ob, self._entity_class)

        self._read_tx = {}

        self._batch_results = None
        self._ef_search = None

    def _ensure_read_tx(self):
        """ Ensures a read TX is created for the current thread. """
        thread_id = threading.get_ident()
        if thread_id in self._read_tx:
            return
        print(f"[objectbox] Beginning read TX for thread {thread_id}...")
        self._read_tx[thread_id] = obx_txn_read(self._ob._c_store)

    def _ensure_no_read_tx(self):
        """ Ensures no read TX exists. """
        thread_ids = self._read_tx.keys()
        for thread_id in thread_ids:
            print(f"[objectbox] Ending read TX for thread {thread_id}... ")
            obx_txn_close(self._read_tx[thread_id])
            del self._read_tx[thread_id]

    def done(self):
        self._ensure_no_read_tx()

    def fit(self, x: np.array) -> None:
        self._ensure_no_read_tx()

        BATCH_SIZE = 10000

        num_objects, vector_dim = x.shape

        if self._box.count() != num_objects:
            self._box.remove_all()
            for i in range(0, num_objects, BATCH_SIZE):
                self._box.put(*[self._entity_class(vector=vector) for vector in x[i:i + BATCH_SIZE]])
                print(f"[objectbox] Inserted {i + BATCH_SIZE} objects...")
            assert self._box.count() == num_objects
        else:
            print(f"[objectbox] DB was already filled!")

    def set_query_arguments(self, ef: float):
        print(f"[objectbox] Query search params; EF: {ef}")
        self._ef_search = ef

    def query(self, q: np.array, n: int) -> np.array:
        query = self._box.query(self._vector_property.nearest_neighbor(q, self._ef_search)).build()
        query.limit(n)

        return np.array([id_ for id_, _ in query.find_ids_with_scores()]) - 1  # Because OBX IDs start at 1

    def batch_query(self, q_batch: np.array, n: int) -> None:
        print(f"[objectbox] Query batch shape: {q_batch.shape}; N: {n}")

        def _run_batch_query(q: np.ndarray):
            self._ensure_read_tx()
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
