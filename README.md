# DynamoDB Vector Search -- Parrot Shop Demo

A step-by-step demo of adding native vector search to DynamoDB, using a
60-item "parrot shop" product catalogue as the running example. Built for a
YouTube tutorial.

1. Start with a plain DynamoDB table (no vectors, no Bedrock).
2. Embed product descriptions with Cohere Embed on Bedrock, in one batched call.
3. Add a vector index to the *existing* table and watch DynamoDB backfill it for free.
4. Run semantic searches against it, scoped by marketplace and filtered inline.

## Prerequisites

- Python 3.9+ with `boto3` installed (`pip install boto3`)
- AWS credentials configured (`aws configure` or environment variables) with
  permission to create/update DynamoDB tables and invoke Bedrock models
- In the Bedrock console: **Model access -> Cohere Embed English v3** enabled
  for your account/region
- A region where DynamoDB vector search is available (the scripts default to
  `us-east-1`)

## Files

| File | Purpose |
|---|---|
| `parrot-products.json` | Source data -- 60 parrot-shop products across the `US` and `EU` marketplaces |
| `step1_create_and_load.py` | Creates the `ParrotShop` table and loads the catalogue. No embeddings yet. |
| `step2_cohere_batched.py` | Embeds every product's Name + Description with Cohere Embed (batched, up to 96 texts/call) and writes the vectors back to each item |
| `step3_add_vector_index.py` | Adds a vector index to the already-populated table and polls until DynamoDB finishes backfilling it |
| `step4_search.py` | Runs semantic searches against the index -- a scripted demo, or a single ad-hoc query |

## Running the demo in order

```bash
# 1. Plain table, no vectors -- the starting point most people are actually in
python step1_create_and_load.py --region us-east-1

# 2. Add embeddings via Cohere Embed on Bedrock (one batched call for all 60 items)
python step2_cohere_batched.py --region us-east-1

# 3. Add the vector index and watch it backfill from the vectors already on the items
python step3_add_vector_index.py --region us-east-1

# 4. Search
python step4_search.py --region us-east-1
```

## Searching

```bash
python step4_search.py --region us-east-1
```

Or run a single query:

```bash
python step4_search.py --region us-east-1 \
  --query "toys for a bored parrot" \
  --marketplace EU \
  --category toys \
  --price-tier mid \
  --top-k 5
```

`--marketplace` is required on every search because `Marketplace` is the
index's hash key. `--category` and `--price-tier` are optional inline filters.
