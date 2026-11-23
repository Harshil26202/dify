import uuid

from dify_vdb_elasticsearch.elasticsearch_vector import ElasticSearchConfig, ElasticSearchVector

from core.rag.datasource.vdb.vector_integration_test_support import (
    AbstractVectorTest,
)
from core.rag.models.document import Document


class ElasticSearchVectorTest(AbstractVectorTest):
    def __init__(self):
        super().__init__()
        self.attributes = ["doc_id", "dataset_id", "document_id", "doc_hash"]
        self.vector = ElasticSearchVector(
            index_name=self.collection_name.lower(),
            config=ElasticSearchConfig(
                use_cloud=False, host="http://localhost", port="9200", username="elastic", password="elastic"
            ),
            attributes=self.attributes,
        )


def test_elasticsearch_vector(setup_mock_redis):
    ElasticSearchVectorTest().run_all_tests()


def test_elasticsearch_delete_by_metadata_field_matches_whole_values(setup_mock_redis):
    test = ElasticSearchVectorTest()
    vector = test.vector
    assert isinstance(vector, ElasticSearchVector)
    app_id = str(uuid.uuid4())
    # Two annotation ids that only share their third group, as UUID4s often do.
    kept_annotation_id = "7c0e2f3a-1111-4b2d-8f00-0123456789ab"
    deleted_annotation_id = "1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"
    annotation_ids = [kept_annotation_id, deleted_annotation_id] + [str(uuid.uuid4()) for _ in range(11)]
    documents = [
        Document(
            page_content=f"question {index}",
            metadata={"doc_id": str(uuid.uuid4()), "annotation_id": annotation_id, "app_id": app_id},
        )
        for index, annotation_id in enumerate(annotation_ids)
    ]
    vector.create(texts=documents, embeddings=[test.example_embedding] * len(documents))
    try:
        vector.delete_by_metadata_field("annotation_id", deleted_annotation_id)
        assert vector._client.count(index=vector._collection_name)["count"] == len(documents) - 1

        vector.delete_by_metadata_field("app_id", app_id)
        assert vector._client.count(index=vector._collection_name)["count"] == 0
    finally:
        vector.delete()
