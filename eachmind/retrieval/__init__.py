"""Retrieval module — pluggable search for encoded memories."""
from eachmind.retrieval.base import Retriever
from eachmind.retrieval.tfidf import TFIDFRetriever

__all__ = ["Retriever", "TFIDFRetriever"]
