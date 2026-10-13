"""Writes the tasks' fixed data into tasks/*/starter. Run once; the output is committed."""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
rng = random.Random(20261013)

KINDS = {
    "headphones": "wireless noise cancelling headphones bluetooth battery earcups audio",
    "kettle": "electric kettle boils water stainless steel litre temperature",
    "backpack": "hiking backpack waterproof straps laptop sleeve travel",
    "lamp": "desk lamp dimmable warm light reading bedside",
    "mug": "ceramic coffee mug dishwasher handle insulated",
    "keyboard": "mechanical keyboard switches backlit typing gaming",
    "tent": "camping tent waterproof poles sleeps outdoor",
    "blender": "kitchen blender smoothie blades motor jug",
    "chair": "office chair ergonomic lumbar support armrests",
    "bottle": "water bottle insulated stainless steel leakproof",
}
STYLES = "compact deluxe classic portable premium everyday studio travel pro lite".split()


def products():
    rows = []
    for i in range(200):
        kind = rng.choice(list(KINDS))
        words = KINDS[kind].split()
        description = " ".join(rng.sample(words, 5) + rng.sample(STYLES, 1))
        rows.append({"id": 1000 + i, "name": f"{rng.choice(STYLES).title()} {kind}", "description": description})
    return "\n".join(json.dumps(r) for r in rows) + "\n"


DOCS = {
    "returns.md": [
        "# Returns",
        "You can return most items within 30 days of delivery for a full refund to the original payment method.",
        "Damaged items: if something arrives damaged, send a photo within 7 days and we refund it, no return needed.",
        "Opened software, gift cards and personalised goods can't be returned.",
    ],
    "shipping.md": [
        "# Shipping",
        "Standard shipping takes three to five working days; express arrives the next working day if ordered by noon.",
        "We ship to every country in the European Union and to the United Kingdom, Norway and Switzerland.",
        "Tracking numbers arrive by email once the parcel leaves the warehouse.",
    ],
    "warranty.md": [
        "# Warranty",
        "Every product carries a two year warranty against manufacturing faults.",
        "Batteries are covered for one year, and wear from normal use isn't covered.",
        "To claim, open a ticket with the order number and a description of the fault.",
    ],
    "accounts.md": [
        "# Accounts",
        "Reset a forgotten password from the sign-in page; the link in the email works for one hour.",
        "Two-factor authentication can be turned on under security settings, with an authenticator app.",
        "Deleting an account removes order history after the legal retention period.",
    ],
    "payments.md": [
        "# Payments",
        "We accept cards, bank transfers and invoices for business customers.",
        "Invoices are payable within 14 days; late invoices pause new orders.",
        "Prices include VAT for customers in the European Union.",
    ],
}

ISSUES = [
    "password reset email never arrives",
    "parcel tracking number shows no updates",
    "kettle stopped heating after two weeks",
    "charged twice for the same order",
    "headphones battery drains overnight",
    "cannot sign in after enabling two factor authentication",
    "invoice address is wrong on business order",
    "refund for damaged blender still missing",
    "express delivery arrived three days late",
    "backpack zipper broke on first trip",
    "discount code rejected at checkout",
    "desk lamp flickers when dimmed",
]
FILLER = "please urgently thanks again today still really kindly".split()


def tickets():
    lines = []
    for issue in ISSUES:
        for _ in range(rng.randint(2, 6)):
            words = issue.split()
            if rng.random() < 0.5:
                words.insert(rng.randrange(len(words) + 1), rng.choice(FILLER))
            lines.append(" ".join(words).capitalize())
    vocabulary = sorted({w for kind in KINDS.values() for w in kind.split()} | set(STYLES))
    while len(lines) < 500:
        lines.append(" ".join(rng.sample(vocabulary, 6)).capitalize())
    rng.shuffle(lines)
    return "\n".join(lines) + "\n"


def main():
    (HERE / "tasks/similar/starter").mkdir(parents=True, exist_ok=True)
    (HERE / "tasks/similar/starter/products.jsonl").write_text(products())
    docs = HERE / "tasks/retrieve/starter/docs"
    docs.mkdir(parents=True, exist_ok=True)
    for name, paragraphs in DOCS.items():
        (docs / name).write_text("\n\n".join(paragraphs) + "\n")
    (HERE / "tasks/dedupe/starter").mkdir(parents=True, exist_ok=True)
    (HERE / "tasks/dedupe/starter/tickets.txt").write_text(tickets())


if __name__ == "__main__":
    main()
