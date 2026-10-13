`products.jsonl` has a product per line: `{"id": ..., "name": ..., "description": ...}`. Build `similar.py`:

- `python similar.py index` builds a search index of the products from their name and description, kept in the current folder.
- `python similar.py like PRODUCT_ID [-k N]` prints the N products most similar to that product (default 5), nearest first, one id per line, never the product itself.

`like` runs in a new process each time and has to be fast: it must not re-embed all the products. `embed.py` here is the embedding model to use: `from embed import embed`.
