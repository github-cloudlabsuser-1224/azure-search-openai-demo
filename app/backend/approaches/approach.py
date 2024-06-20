import os
from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    List,
    Optional,
    TypedDict,
    cast,
)
from urllib.parse import urljoin

import aiohttp
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from core.authentication import AuthenticationHelper
from text import nonewlines


@dataclass
class Document:
    id: Optional[str]
    content: Optional[str]
    embedding: Optional[List[float]]
    image_embedding: Optional[List[float]]
    category: Optional[str]
    sourcepage: Optional[str]
    sourcefile: Optional[str]
    oids: Optional[List[str]]
    groups: Optional[List[str]]
    captions: List[QueryCaptionResult]
    score: Optional[float] = None
    reranker_score: Optional[float] = None

    def serialize_for_results(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": Document.trim_embedding(self.embedding),
            "imageEmbedding": Document.trim_embedding(self.image_embedding),
            "category": self.category,
            "sourcepage": self.sourcepage,
            "sourcefile": self.sourcefile,
            "oids": self.oids,
            "groups": self.groups,
            "captions": (
                [
                    {
                        "additional_properties": caption.additional_properties,
                        "text": caption.text,
                        "highlights": caption.highlights,
                    }
                    for caption in self.captions
                ]
                if self.captions
                else []
            ),
            "score": self.score,
            "reranker_score": self.reranker_score,
        }

    @classmethod
    def trim_embedding(cls, embedding: Optional[List[float]]) -> Optional[str]:
        """Returns a trimmed list of floats from the vector embedding."""
        if embedding:
            if len(embedding) > 2:
                # Format the embedding list to show the first 2 items followed by the count of the remaining items."""
                return f"[{embedding[0]}, {embedding[1]} ...+{len(embedding) - 2} more]"
            else:
                return str(embedding)

        return None


@dataclass
class ThoughtStep:
    title: str
    description: Optional[Any]
    props: Optional[dict[str, Any]] = None


from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict
from abc import ABC
from azure.search.documents import SearchClient, Document
from azure.core.async_paging import AsyncItemPaged
from openai import AsyncOpenAI
from urllib.parse import urljoin
import aiohttp
import os

class Approach(ABC):
    """
    Base class for different approaches used in the application.
    """

    def __init__(
        self,
        search_client: SearchClient,
        openai_client: AsyncOpenAI,
        auth_helper: AuthenticationHelper,
        query_language: Optional[str],
        query_speller: Optional[str],
        embedding_deployment: Optional[str],
        embedding_model: str,
        embedding_dimensions: int,
        openai_host: str,
        vision_endpoint: str,
        vision_token_provider: Callable[[], Awaitable[str]],
    ):
        """
        Initialize the Approach class.

        Args:
            search_client (SearchClient): The Azure Cognitive Search client.
            openai_client (AsyncOpenAI): The OpenAI client.
            auth_helper (AuthenticationHelper): The authentication helper.
            query_language (Optional[str]): The query language.
            query_speller (Optional[str]): The query speller.
            embedding_deployment (Optional[str]): The embedding deployment.
            embedding_model (str): The embedding model.
            embedding_dimensions (int): The embedding dimensions.
            openai_host (str): The OpenAI host.
            vision_endpoint (str): The vision endpoint.
            vision_token_provider (Callable[[], Awaitable[str]]): The token provider for vision endpoint.
        """
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.query_language = query_language
        self.query_speller = query_speller
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.openai_host = openai_host
        self.vision_endpoint = vision_endpoint
        self.vision_token_provider = vision_token_provider

    def build_filter(self, overrides: Dict[str, Any], auth_claims: Dict[str, Any]) -> Optional[str]:
        """
        Build the filter based on the overrides and authentication claims.

        Args:
            overrides (Dict[str, Any]): The overrides for the filter.
            auth_claims (Dict[str, Any]): The authentication claims.

        Returns:
            Optional[str]: The built filter string.
        """
        exclude_category = overrides.get("exclude_category")
        security_filter = self.auth_helper.build_security_filters(overrides, auth_claims)
        filters = []
        if exclude_category:
            filters.append("category ne '{}'".format(exclude_category.replace("'", "''")))
        if security_filter:
            filters.append(security_filter)
        return None if len(filters) == 0 else " and ".join(filters)

    async def search(
        self,
        top: int,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
        use_text_search: bool,
        use_vector_search: bool,
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
        minimum_search_score: Optional[float],
        minimum_reranker_score: Optional[float],
    ) -> List[Document]:
        """
        Perform a search operation.

        Args:
            top (int): The number of documents to retrieve.
            query_text (Optional[str]): The query text.
            filter (Optional[str]): The filter string.
            vectors (List[VectorQuery]): The vector queries.
            use_text_search (bool): Flag indicating whether to use text search.
            use_vector_search (bool): Flag indicating whether to use vector search.
            use_semantic_ranker (bool): Flag indicating whether to use semantic ranker.
            use_semantic_captions (bool): Flag indicating whether to use semantic captions.
            minimum_search_score (Optional[float]): The minimum search score.
            minimum_reranker_score (Optional[float]): The minimum reranker score.

        Returns:
            List[Document]: The list of retrieved documents.
        """
        search_text = query_text if use_text_search else ""
        search_vectors = vectors if use_vector_search else []
        if use_semantic_ranker:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector_queries=search_vectors,
                query_type=QueryType.SEMANTIC,
                query_language=self.query_language,
                query_speller=self.query_speller,
                semantic_configuration_name="default",
                semantic_query=query_text,
            )
        else:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                vector_queries=search_vectors,
            )

        documents = []
        async for page in results.by_page():
            async for document in page:
                documents.append(
                    Document(
                        id=document.get("id"),
                        content=document.get("content"),
                        embedding=document.get("embedding"),
                        image_embedding=document.get("imageEmbedding"),
                        category=document.get("category"),
                        sourcepage=document.get("sourcepage"),
                        sourcefile=document.get("sourcefile"),
                        oids=document.get("oids"),
                        groups=document.get("groups"),
                        captions=cast(List[QueryCaptionResult], document.get("@search.captions")),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                    )
                )

            qualified_documents = [
                doc
                for doc in documents
                if (
                    (doc.score or 0) >= (minimum_search_score or 0)
                    and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
                )
            ]

        return qualified_documents

    def get_sources_content(
        self, results: List[Document], use_semantic_captions: bool, use_image_citation: bool
    ) -> List[str]:
        """
        Get the content of the sources.

        Args:
            results (List[Document]): The list of documents.
            use_semantic_captions (bool): Flag indicating whether to use semantic captions.
            use_image_citation (bool): Flag indicating whether to use image citation.

        Returns:
            List[str]: The list of source contents.
        """
        if use_semantic_captions:
            return [
                (self.get_citation((doc.sourcepage or ""), use_image_citation))
                + ": "
                + nonewlines(" . ".join([cast(str, c.text) for c in (doc.captions or [])]))
                for doc in results
            ]
        else:
            return [
                (self.get_citation((doc.sourcepage or ""), use_image_citation)) + ": " + nonewlines(doc.content or "")
                for doc in results
            ]

    def get_citation(self, sourcepage: str, use_image_citation: bool) -> str:
        """
        Get the citation for a source.

        Args:
            sourcepage (str): The source page.
            use_image_citation (bool): Flag indicating whether to use image citation.

        Returns:
            str: The citation string.
        """
        if use_image_citation:
            return sourcepage
        else:
            path, ext = os.path.splitext(sourcepage)
            if ext.lower() == ".png":
                page_idx = path.rfind("-")
                page_number = int(path[page_idx + 1 :])
                return f"{path[:page_idx]}.pdf#page={page_number}"

            return sourcepage

    async def compute_text_embedding(self, q: str):
        """
        Compute the text embedding.

        Args:
            q (str): The input text.

        Returns:
            VectorizedQuery: The vectorized query.
        """
        SUPPORTED_DIMENSIONS_MODEL = {
            "text-embedding-ada-002": False,
            "text-embedding-3-small": True,
            "text-embedding-3-large": True,
        }

        class ExtraArgs(TypedDict, total=False):
            dimensions: int

        dimensions_args: ExtraArgs = (
            {"dimensions": self.embedding_dimensions} if SUPPORTED_DIMENSIONS_MODEL[self.embedding_model] else {}
        )
        embedding = await self.openai_client.embeddings.create(
            model=self.embedding_deployment if self.embedding_deployment else self.embedding_model,
            input=q,
            **dimensions_args,
        )
        query_vector = embedding.data[0].embedding
        return VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="embedding")

    async def compute_image_embedding(self, q: str):
        """
        Compute the image embedding.

        Args:
            q (str): The input text.

        Returns:
            VectorizedQuery: The vectorized query.
        """
        endpoint = urljoin(self.vision_endpoint, "computervision/retrieval:vectorizeText")
        headers = {"Content-Type": "application/json"}
        params = {"api-version": "2023-02-01-preview", "modelVersion": "latest"}
        data = {"text": q}

        headers["Authorization"] = "Bearer " + await self.vision_token_provider()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=endpoint, params=params, headers=headers, json=data, raise_for_status=True
            ) as response:
                json = await response.json()
                image_query_vector = json["vector"]
        return VectorizedQuery(vector=image_query_vector, k_nearest_neighbors=50, fields="imageEmbedding")

    async def run(
        self,
        messages: List[ChatCompletionMessageParam],
        session_state: Any = None,
        context: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """
        Run the approach.

        Args:
            messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
            session_state (Any): The session state.
            context (Dict[str, Any]): The context.

        Returns:
            Dict[str, Any]: The response from the approach.
        """
        raise NotImplementedError

    async def run_stream(
        self,
        messages: List[ChatCompletionMessageParam],
        session_state: Any = None,
        context: Dict[str, Any] = {},
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the approach in a streaming fashion.

        Args:
            messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
            session_state (Any): The session state.
            context (Dict[str, Any]): The context.

        Yields:
            Dict[str, Any]: The response from the approach.
        """
        raise NotImplementedError
