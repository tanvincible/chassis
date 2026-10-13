Build a command-line notes app, `notes.py`, that finds notes by meaning:

- `python notes.py add "TEXT"` stores a note and prints its id (an integer).
- `python notes.py search "QUERY" [-k N]` prints the N notes closest in meaning (default 3), nearest first, one per line as `ID<TAB>TEXT`.
- `python notes.py delete ID` deletes a note; it never shows up in a search again.

Notes have to survive between runs: each command is a separate process. Keep everything the app stores in the current folder.

`embed.py` here is the embedding model to use: `from embed import embed`.
