from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
import os
import pretty_errors
# install smalldocling, a vison LM for document OCR.
# read all documents from the data directory.
# split the documents based on file extension.
# load each group based on the file extension.
# chunk the groups and add them to the Qdrant database.
# for documents that have images, use the smalldocling model to extract text from the images or use llm model from ollama to extract information
# from the documents.
# insert the emveddings into the Qdrant database.
