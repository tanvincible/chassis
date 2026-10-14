# Style

This is how the Chassis docs are written, so that every page sounds like one project and gets a
reader, or an agent, to working code quickly. It covers the README, the book, the Python docs, the
package READMEs, docstrings, error messages and the changelog. Thank you for following it; a pull
request that does needs fewer rounds of review.

## Tone

Write the way [CONTRIBUTING](CONTRIBUTING.md) does: warm and welcoming, and firm. Talk to the reader
as "you". Say plainly what they can do and what Chassis won't do, without apologizing for either.

- No marketing: not "high-performance", "blazing", "seamless" or "raw speed of Rust". Say what
  Chassis does, and let the numbers speak.
- No hedging: when something is true, say it. When it isn't known, say that, or leave it out.
- Say where Chassis trails. Readers trust a page that names its limits.

## Claims

Every claim about speed, size or recall is measured, and says on what: the machine, the data and
the date, with a link to how to reproduce it. "On an Apple M5, with 99,000 vectors of 1,536
dimensions not yet in memory, the first result came 0.11 s after the process started", not "opens
instantly". A claim nobody has measured goes.

## Sentences

One idea per sentence. When a sentence chains clauses until the reader has to go back for its
subject, split it:

> ✗ `search` takes `&self`, so many threads can search one `VectorIndex` at once, as in an
> `Arc<RwLock<VectorIndex>>` where searches take the read lock and `add`, `delete` and `flush`,
> which take `&mut self`, the write lock.
>
> ✓ `search` takes `&self`, so many threads can search one `VectorIndex` at once. Wrap it in an
> `Arc<RwLock<VectorIndex>>`: searches take the read lock, and `add`, `delete` and `flush`, which
> take `&mut self`, take the write lock.

Sentences of 15 to 20 words on average read well. A list reads better than a sentence with seven
commas.

## Terms

- **The standard term first, then plain words.** People and agents search for the standard names:
  "8-bit quantization (LVQ)", then what it does. Plain words alone hide the page from them.
- **id** and **ids**, never "ID" or "Id": the API says `id`.
- **indexes**, not "indices".
- **distance**, not "score": smaller is nearer.
- **index**, not "database": Chassis is an embedded vector index. It keeps vectors and ids; your
  text and metadata stay in your own store.

## Headings

One or two words, never a sentence, unless no word or two make sense. Use sentence case:
"Filtered search", not "Filtered Search". A function's name is a fine heading. Page titles and
sidebar names are headings too.

## Pages

Open with what the reader will be able to do, then get to it. A caveat goes after the quick start,
not before it. Link to the page or decision record that has the details instead of repeating them.

## Examples

Examples run as written: imports included, no helper left undefined. Agents copy them as they are.
When code stands for something the reader brings, say so, as
[Next to Your Database](docs/src/guide/with-a-database.md) does: "`embed(texts)` stands for your
embedding model."

## Placement

- **README**: the front door. What Chassis is, a few lines of use, and links. No tables of numbers
  or decision records.
- **The book and the Python docs**: how to use Chassis, and how it works.
- **Decision records** (`docs/adr`): why each choice was made, with its measurements. They are not
  published on the site.
- **CHANGELOG**: what changed for users, one plain line each: fixes and behaviour changes, never
  the experiment that found them.
- **Errors**: one line saying what was wrong, with the values, then a `help:` line saying what to
  do instead, in words that fit every language. Each one is listed in
  [Errors](docs/src/guide/errors.md).
