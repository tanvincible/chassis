# langchain-chassisdb

A [LangChain](https://python.langchain.com) vector store on
[Chassis](https://github.com/tanvincible/chassis): semantic search over your documents, kept in two
files, inside your own process. No server to run, nothing to connect to.

```sh
pip install langchain-chassisdb
```

```python
from langchain_chassisdb import ChassisVectorStore
from langchain_openai import OpenAIEmbeddings  # or any LangChain embeddings

store = ChassisVectorStore("docs.chassis", embedding=OpenAIEmbeddings())
store.add_texts(
    ["Refunds take five working days.", "We ship to the EU and the UK."],
    metadatas=[{"topic": "billing"}, {"topic": "shipping"}],
)

retriever = store.as_retriever(search_kwargs={"k": 4, "filter": {"topic": "billing"}})
retriever.invoke("how long does a refund take?")
```

- **Files**: `docs.chassis` holds the vectors and `docs.db` the documents and metadata. Open the
  same path again, with the same embeddings, to pick up where you left off. Everything added or
  deleted is saved when the call returns. A call that a crash or an error stops changes neither
  file: the store brings its vectors back in step when it is next opened or written.
- **Ids**: give your own with `ids=`; adding a document with an id that is stored replaces it.
  `get_by_ids` and `delete` take them.
- **Filters**: `filter={"key": value, ...}` keeps documents whose metadata has every key with that
  value.
- **Scores**: `similarity_search_with_score` gives distances (smaller is closer);
  `similarity_search_with_relevance_scores` gives 0 to 1. `max_marginal_relevance_search` picks
  documents that are close to the query and different from each other.
- **Options**: `metric="cosine"` (the default) or `"euclidean"`; `precision="half"` keeps vectors
  in about half the space, with the same recall on the embeddings measured. They're fixed when the
  store is created: open it again with the same ones, or an `OptionsMismatchError` says which to
  pass. A `read_only` store takes them from the file.
- **More processes**: one process writes a store at a time. Others open it with
  `read_only=True`, take no lock, and see what the writer last saved.

It passes LangChain's standard tests for vector stores, sync and async.
