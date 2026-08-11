"""
Demo searches against the ParrotShop vector index, using Cohere Embed.

Must match how the corpus was embedded in step2_cohere_batched.py:
  * same model (cohere.embed-english-v3)
  * same dimensions (1024)
  * but input_type="search_query" instead of "search_document"

That last difference is deliberate, not a bug. Cohere prepends different
special tokens for queries and documents, and using search_document for both
quietly degrades relevance rather than raising an error.

    python step4_search.py --region us-east-1
    python step4_search.py --region us-east-1 --query "toys for a bored parrot" --marketplace EU
"""

import argparse
import json

import boto3
from botocore.config import Config

TABLE = "ParrotShop"
INDEX = "DescriptionIndex"
MODEL_ID = "cohere.embed-english-v3"


def make_clients(region):
    ddb = boto3.client("dynamodb", region_name=region)
    bedrock = boto3.client(
        "bedrock-runtime", region_name=region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    return ddb, bedrock


def embed_query(bedrock, text):
    """search_query, NOT search_document -- the corpus used search_document."""
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "texts": [text[:2048]],
            "input_type": "search_query",
        }),
    )
    return json.loads(response["body"].read())["embeddings"][0]


def search(ddb, bedrock, query, marketplace,
           category=None, price_tier=None, top_k=5):
    embedding = embed_query(bedrock, query)

    # Query vector: plain list of N values -- no 'L' wrapper.
    # This differs from the stored attribute format.
    search_vector = [{"N": str(v)} for v in embedding]

    print(f"\nSearching for '{embedding}' in {marketplace}")

    # Marketplace is the index HASH key, so it is required on every search.
    conditions = ["Marketplace = :m"]
    values = {":m": {"S": marketplace}}

    if category:
        conditions.append("Category = :c")
        values[":c"] = {"S": category}
    if price_tier:
        conditions.append("PriceTier = :t")
        values[":t"] = {"S": price_tier}

    return ddb.search_vectors(
        TableName=TABLE,
        IndexName=INDEX,
        SearchVector=search_vector,
        TopK=top_k,
        SearchConditionExpression=" AND ".join(conditions),
        ExpressionAttributeValues=values,
        # Vector excluded by default -- returned bytes are what you pay for.
        ProjectionExpression="ProductId, #n, Category, Price, Currency",
        ExpressionAttributeNames={"#n": "Name"},
        ReturnConsumedCapacity="INDEXES",
    )


def show(response, label):
    print(f"\n{label}")
    print("-" * min(len(label), 70))
    for rank, result in enumerate(response["SearchResults"], 1):
        item = result["Item"]
        print(f"  {rank}. {item['Name']['S']}")
        print(f"     {item['Currency']['S']} {item['Price']['N']}"
              f"  |  {item['Category']['S']}"
              f"  |  score {result.get('Score'):.4f}")
    cc = response.get("ConsumedCapacity", {})
    print(f"  [VectorSearchRequestBytes: {cc.get('VectorSearchRequestBytes')}]")


def run_demo(ddb, bedrock):
    # 1. Partition scoping -- same query, different inventory
    q = "something to keep my parrot busy while I am at work"
    show(search(ddb, bedrock, q, "US"), f'US  <- "{q}"')
    show(search(ddb, bedrock, q, "EU"), f'EU  <- "{q}"')

    # 2. Semantic match with no shared keywords
    q = "my bird is losing feathers and looking scruffy"
    show(search(ddb, bedrock, q, "US"), f'US  <- "{q}"')

    # 3. Inline filter narrowing
    q = "natural wood for my bird to stand on"
    show(search(ddb, bedrock, q, "EU"), f'EU, no filter  <- "{q}"')
    show(search(ddb, bedrock, q, "EU", category="perches"),
         f'EU, Category=perches  <- "{q}"')

    # 4. Why PriceTier exists (inline filters are equality-only)
    q = "a large cage for a macaw"
    show(search(ddb, bedrock, q, "US", price_tier="premium"),
         f'US, PriceTier=premium  <- "{q}"')

    # 5. Top-K always fills, even with nothing relevant
    q = "tropical fish aquarium filter"
    show(search(ddb, bedrock, q, "US"),
         f'US  <- "{q}"  (nothing matches -- judge by Score)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--query", help="Run a single query instead of the demo")
    ap.add_argument("--marketplace", default="US", choices=["US", "EU"])
    ap.add_argument("--category")
    ap.add_argument("--price-tier", choices=["budget", "mid", "premium"])
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    ddb, bedrock = make_clients(args.region)

    if args.query:
        response = search(ddb, bedrock, args.query, args.marketplace,
                          args.category, args.price_tier, args.top_k)
        show(response, f'{args.marketplace}  <- "{args.query}"')
    else:
        run_demo(ddb, bedrock)


if __name__ == "__main__":
    main()
