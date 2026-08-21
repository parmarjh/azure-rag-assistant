from __future__ import annotations

from ..models import Chunk, Retrieved


class AzureSearchIndex:
    """Azure AI Search adapter using the 11.5+ vector query API."""

    def __init__(self, endpoint, index_name, credential=None, api_key=None):
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            from azure.search.documents.models import VectorizedQuery
        except ImportError as exc:
            raise ImportError("Install the azure extra to use Azure Search") from exc
        if credential is None and api_key:
            credential = AzureKeyCredential(api_key)
        self.client = SearchClient(endpoint, index_name, credential)
        self._vector_query = VectorizedQuery

    def create(self):
        return None

    def upload(self, chunks):
        self.client.upload_documents([self._serialize(c) for c in chunks])

    def _serialize(self, c):
        value = dict(c.__dict__)
        value["id"] = c.chunk_id
        value["content_vector"] = c.embedding
        value.pop("embedding", None)
        return value

    def search(self, query, vector, filters, top_k):
        kwargs = {"search_text": query, "top": top_k}
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
                output.append(Retrieved(chunk, float(row.get("@search.score", 0))))
            except KeyError:
                continue
        return output
