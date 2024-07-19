from transformers.agents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"

    def __init__(self, vectordb, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        assert kwargs['k']
        self.k = kwargs['k']
        self.metadata = []

    def reset_metadata(self) -> None:
        self.metadata = []

    def forward(self, query: str) -> str:
        assert isinstance(query, str), f"Your search query must be a string: '{query}'"

        docs = self.vectordb.similarity_search(
            query,
            k=self.k,
            # todo filter={'id': '/Users/johnday/repos/md/bob/bob_omnia_assessment_Day2_Performance/bob_omnia_assessment_Day2_Performance.md'},
        )
        self.metadata = [doc.metadata for doc in docs]

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

