`docs/` holds Markdown files. Write `retrieve.py` for a RAG pipeline, with two functions:

- `index_docs(folder)` splits each file into paragraphs (blocks separated by a blank line) and indexes them, keeping the index in the current folder. Running it again on changed files must not leave stale paragraphs behind.
- `retrieve(query, k=4)` returns the k paragraphs closest in meaning, nearest first, as dicts `{"file": <file name>, "text": <paragraph>, "distance": <float>}`. It works in a new process, without indexing again.

`embed.py` here is the embedding model to use: `from embed import embed`.
