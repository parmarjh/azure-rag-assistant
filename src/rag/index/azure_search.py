from __future__ import annotations

from datetime import datetime

from ..models import Chunk, Retrieved


class AzureSearchIndex:
    """Azure AI Search adapter using the 11.5+ vector query API."""

    def __init__(self, endpoint, index_name, credential=None, api_key=None):
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.identity import DefaultAzureCredential
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient
            from azure.search.documents.indexes.models import (
                HnswAlgorithmConfiguration,
                SearchableField,
                SearchField,
                SearchFieldDataType,
                SearchIndex,
                SemanticConfiguration,
                SemanticField,
                SemanticPrioritizedFields,
                SemanticSearch,
                SimpleField,
                VectorSearch,
                VectorSearchProfile,
            )
            from azure.search.documents.models import VectorizedQuery
        except ImportError as exc:
            raise ImportError("Install the azure extra to use Azure Search") from exc
        if credential is None and api_key:
            credential = AzureKeyCredential(api_key)
        if credential is None:
            credential = DefaultAzureCredential()
        self._models = {
            "HnswAlgorithmConfiguration": HnswAlgorithmConfiguration,
            "SearchField": SearchField,
            "SearchFieldDataType": SearchFieldDataType,
            "SearchIndex": SearchIndex,
            "SearchableField": SearchableField,
            "SemanticConfiguration": SemanticConfiguration,
            "SemanticField": SemanticField,
            "SemanticPrioritizedFields": SemanticPrioritizedFields,
            "SemanticSearch": SemanticSearch,
            "SimpleField": SimpleField,
            "VectorSearch": VectorSearch,
            "VectorSearchProfile": VectorSearchProfile,
        }
        self.endpoint, self.index_name, self.credential = endpoint, index_name, credential
        self.client = SearchClient(endpoint, index_name, credential)
        self.index_client = SearchIndexClient(endpoint, credential)
        self._vector_query = VectorizedQuery

    def create(self):
        m = self._models
        collection = m["SearchFieldDataType"].Collection(m["SearchFieldDataType"].String)
        vector_type = m["SearchFieldDataType"].Collection(m["SearchFieldDataType"].Single)
        fields = [
            m["SimpleField"](name="id", type=m["SearchFieldDataType"].String, key=True),
            m["SearchableField"](name="content", type=m["SearchFieldDataType"].String),
            m["SearchableField"](name="header", type=m["SearchFieldDataType"].String),
            m["SearchField"](name="content_vector", type=vector_type, searchable=True,
                             vector_search_dimensions=3072, vector_search_profile_name="default"),
            m["SimpleField"](name="doc_id", type=m["SearchFieldDataType"].String, filterable=True),
            m["SearchableField"](name="doc_title", type=m["SearchFieldDataType"].String),
            m["SimpleField"](name="filename", type=m["SearchFieldDataType"].String, filterable=True),
            m["SimpleField"](name="source_uri", type=m["SearchFieldDataType"].String),
            m["SimpleField"](name="department", type=m["SearchFieldDataType"].String,
                             filterable=True, facetable=True),
            m["SearchField"](name="security_groups", type=collection, filterable=True),
            m["SimpleField"](name="doc_type", type=m["SearchFieldDataType"].String, filterable=True),
            m["SimpleField"](name="doc_family", type=m["SearchFieldDataType"].String, filterable=True),
            m["SimpleField"](name="version", type=m["SearchFieldDataType"].String),
            m["SimpleField"](name="effective_date", type=m["SearchFieldDataType"].DateTimeOffset,
                             filterable=True, sortable=True),
            m["SimpleField"](name="expiry_date", type=m["SearchFieldDataType"].DateTimeOffset,
                             filterable=True, sortable=True),
            m["SimpleField"](name="is_current", type=m["SearchFieldDataType"].Boolean,
                             filterable=True),
            m["SimpleField"](name="supersedes", type=m["SearchFieldDataType"].String),
            m["SimpleField"](name="superseded_by", type=m["SearchFieldDataType"].String),
            m["SearchableField"](name="section_path", type=m["SearchFieldDataType"].String),
            m["SimpleField"](name="section_number", type=m["SearchFieldDataType"].String),
            m["SimpleField"](name="page", type=m["SearchFieldDataType"].Int32),
            m["SimpleField"](name="chunk_index", type=m["SearchFieldDataType"].Int32),
            m["SimpleField"](name="token_count", type=m["SearchFieldDataType"].Int32),
            m["SearchField"](name="keywords", type=collection, searchable=True, filterable=True),
        ]
        vector_search = m["VectorSearch"](
            algorithms=[m["HnswAlgorithmConfiguration"](name="hnsw")],
            profiles=[m["VectorSearchProfile"](name="default", algorithm_configuration_name="hnsw")],
        )
        semantic = m["SemanticSearch"](configurations=[m["SemanticConfiguration"](
            name="default",
            prioritized_fields=m["SemanticPrioritizedFields"](
                title_field=m["SemanticField"](field_name="doc_title"),
                content_fields=[m["SemanticField"](field_name="content")],
                keywords_fields=[m["SemanticField"](field_name="keywords")],
            ),
        )])
        self.index_client.create_or_update_index(m["SearchIndex"](
            name=self.index_name, fields=fields, vector_search=vector_search,
            semantic_search=semantic))

    def upload(self, chunks):
        self.client.upload_documents([self._serialize(c) for c in chunks])

    def _serialize(self, c):
        value = dict(c.__dict__)
        value["id"] = c.chunk_id
        value["content_vector"] = c.embedding
        value.pop("embedding", None)
        for date_field in ("effective_date", "expiry_date"):
            if value.get(date_field):
                try:
                    value[date_field] = datetime.fromisoformat(value[date_field]).isoformat() + "Z"
                except ValueError:
                    value[date_field] = None
        return value

    def search(self, query, vector, filters, top_k):
        kwargs = {"search_text": query, "top": top_k}
        if query:
            kwargs.update(query_type="semantic", semantic_configuration_name="default")
        if vector:
            kwargs["vector_queries"] = [self._vector_query(
                vector=vector, k_nearest_neighbors=top_k, fields="content_vector")]
        clauses = []
        if filters.get("is_current"):
            clauses.append("is_current eq true")
        if filters.get("department"):
            clauses.append(f"department eq '{filters['department']}'")
        if filters.get("groups"):
            clauses.append(f"security_groups/any(g: search.in(g, '{','.join(filters['groups'])}'))")
        if clauses:
            kwargs["filter"] = " and ".join(clauses)
        results = self.client.search(**kwargs)
        output = []
        for row in results:
            data = dict(row)
            data.pop("@search.score", None)
            data.pop("@search.reranker_score", None)
            try:
                chunk = Chunk(
                    chunk_id=data["id"], doc_id=data["doc_id"], doc_title=data["doc_title"],
                    filename=data["filename"], source_uri=data.get("source_uri", ""),
                    department=data["department"], security_groups=data.get("security_groups", []),
                    doc_type=data.get("doc_type", ""), doc_family=data.get("doc_family", ""),
                    version=data.get("version", ""), effective_date=data.get("effective_date"),
                    expiry_date=data.get("expiry_date"), is_current=data.get("is_current", True),
                    supersedes=data.get("supersedes"), superseded_by=data.get("superseded_by"),
                    section_path=data.get("section_path", ""), section_number=data.get("section_number"),
                    page=data.get("page"), chunk_index=data.get("chunk_index", 0),
                    token_count=data.get("token_count", 0), keywords=data.get("keywords", []),
                    header=data.get("header", ""), content=data["content"],
                    embedding=data.get("content_vector", []))
                score = row.get("@search.reranker_score", row.get("@search.score", 0))
                output.append(Retrieved(chunk, float(score or 0)))
            except KeyError:
                continue
        return output
