import subprocess
import sys
import uuid

import weaviate as wc
from weaviate.classes.config import *
from weaviate.classes.query import MetadataQuery
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from ..base.module import BaseANN


class Weaviate(BaseANN):
    def __init__(self, metric, max_connections, ef_construction=512):
        self.class_name = "Vector"
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.distance = {
            "angular": VectorDistances.COSINE,
            "euclidean": VectorDistances.L2_SQUARED,
        }[metric]

        print(f"[weaviate] Started; "
              f"metric: {metric}, "
              f"max_connections: {max_connections}, "
              f"ef_construction: {ef_construction}")

        self.client = wc.connect_to_embedded(version="1.25.0")

    def fit(self, X):
        print(f"[weaviate] Creating collection; "
              f"distance: {self.distance}, "
              f"ef_construction: {self.ef_construction}, "
              f"max_connections: {self.max_connections}")
        collection = self.client.collections.create(
            name=self.class_name,
            properties=[
                Property(name="i", data_type=DataType.INT),
            ],
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=self.distance,
                ef_construction=self.ef_construction,
                max_connections=self.max_connections
            )
        )
        print("[weaviate] Collection created")
        with collection.batch.fixed_size(batch_size=100) as batch:
            for i, x in enumerate(X):
                batch.add_object(
                    properties={"i": i},
                    uuid=uuid.UUID(int=i),
                    vector=x
                )
        print(f"[weaviate] Dataset inserted; Count: {X.shape[0]}")

    def set_query_arguments(self, ef):
        print(f"[weaviate] Setting query argument: ef={ef}")
        self.ef = ef
        collection = self.client.collections.get(self.class_name)

        # TODO You are using the `vector_index_config` argument in the `collection.config.update()` method, which is
        #   deprecated.
        collection.config.update(vector_index_config=Reconfigure.VectorIndex.hnsw(ef=self.ef))

    def query(self, v, n):
        collection = self.client.collections.get(self.class_name)
        query_result = collection.query.near_vector(near_vector=v.tolist(), limit=n, return_properties=["i"])
        return [object_.properties["i"] for object_ in query_result.objects]

    def __str__(self):
        return f"Weaviate(ef={self.ef}, maxConnections={self.max_connections}, efConstruction={self.ef_construction})"
