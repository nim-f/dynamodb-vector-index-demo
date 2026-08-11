"""
STEP 2 (batched) -- Add embeddings using Cohere Embed on Bedrock.

Titan cannot batch: InvokeModel takes one inputText per call, so 60 products
means 60 requests. Cohere Embed accepts up to 96 texts per call, so the same
60 products take ONE request -- which sidesteps request-per-minute throttling
almost entirely.

Two further advantages:
  * Bedrock quotas are per-model, so this draws on a different bucket to Titan
  * cohere.embed-english-v3 outputs 1024 dimensions, exactly like Titan at
    1024 -- so step3_add_vector_index.py needs no change at all

    python step2_cohere_batched.py --region us-east-1

IMPORTANT -- you cannot mix models. Vectors from Titan and Cohere live in
different vector spaces. If some items were already embedded with Titan, this
script re-embeds everything by default so the whole table is consistent.

Requires Cohere Embed model access enabled in Bedrock. That is a separate
grant from Titan: Bedrock console -> Model access -> Cohere Embed English v3.
"""

import argparse
import json
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

TABLE = "ParrotShop"
MODEL_ID = "cohere.embed-english-v3"
DIMENSIONS = 1024          # matches Titan at 1024 -- index config unchanged
MAX_TEXTS_PER_CALL = 96    # hard API limit
MAX_CHARS_PER_TEXT = 2048  # hard API limit


def make_bedrock(region):
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      read_timeout=60),
    )


def embed_batch(bedrock, texts, input_type="search_document"):
    """Embed up to 96 texts in a single call.

    input_type matters for retrieval quality: use search_document for the
    corpus you store, and search_query for queries at search time. Cohere
    prepends different special tokens for each, and mixing them up degrades
    results in a way that is hard to spot.
    """
    if len(texts) > MAX_TEXTS_PER_CALL:
        raise ValueError(f"Cohere accepts at most {MAX_TEXTS_PER_CALL} texts")

    truncated = [t[:MAX_CHARS_PER_TEXT] for t in texts]
    over = sum(1 for a, b in zip(texts, truncated) if a != b)
    if over:
        print(f"  note: {over} text(s) truncated to {MAX_CHARS_PER_TEXT} chars")

    for attempt in range(1, 6):
        try:
            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "texts": truncated,
                    "input_type": input_type,
                }),
            )
            return json.loads(response["body"].read())["embeddings"]

        except ClientError as err:
            code = err.response["Error"]["Code"]
            if code not in ("ThrottlingException", "TooManyRequestsException"):
                raise
            if attempt == 5:
                raise
            wait = 2 ** attempt
            print(f"  throttled, retrying in {wait}s")
            time.sleep(wait)


def scan_all(ddb):
    items, kwargs = [], {"TableName": TABLE}
    while True:
        page = ddb.scan(**kwargs)
        items.extend(page["Items"])
        if "LastEvaluatedKey" not in page:
            return items
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--only-missing", action="store_true",
                    help="Embed only items with no vector. Use ONLY if every "
                         "existing vector already came from this same model.")
    args = ap.parse_args()

    ddb = boto3.client("dynamodb", region_name=args.region)
    bedrock = make_bedrock(args.region)

    items = scan_all(ddb)
    todo = [i for i in items
            if not args.only_missing or "DescriptionVector" not in i]

    print(f"Found {len(items)} items, embedding {len(todo)}.")
    if not args.only_missing:
        print("Re-embedding everything so the whole table uses one model.")

    if not todo:
        print("Nothing to do.")
        return

    texts = [f"{i['Name']['S']}. {i['Description']['S']}" for i in todo]

    calls = (len(texts) + MAX_TEXTS_PER_CALL - 1) // MAX_TEXTS_PER_CALL
    print(f"{len(texts)} texts -> {calls} Bedrock call(s) "
          f"(Titan would need {len(texts)}).\n")

    started = time.monotonic()
    vectors = []
    for start in range(0, len(texts), MAX_TEXTS_PER_CALL):
        chunk = texts[start:start + MAX_TEXTS_PER_CALL]
        print(f"  embedding texts {start + 1}-{start + len(chunk)}...")
        vectors.extend(embed_batch(bedrock, chunk, "search_document"))

    elapsed = time.monotonic() - started
    print(f"\nGot {len(vectors)} vectors in {elapsed:.1f}s.")

    if len(vectors) != len(todo):
        raise RuntimeError(
            f"Expected {len(todo)} vectors, got {len(vectors)}")

    actual_dim = len(vectors[0])
    if actual_dim != DIMENSIONS:
        raise RuntimeError(
            f"Model returned {actual_dim} dimensions, expected {DIMENSIONS}. "
            f"Update DIMENSIONS here and in step3_add_vector_index.py.")
    print(f"Dimensions confirmed: {actual_dim}")

    for item, vector in zip(todo, vectors):
        ddb.update_item(
            TableName=TABLE,
            Key={"ProductId": {"S": item["ProductId"]["S"]}},
            UpdateExpression="SET DescriptionVector = :v",
            ExpressionAttributeValues={
                ":v": {"L": [{"N": str(v)} for v in vector]}
            },
        )
    print(f"Wrote {len(todo)} vectors to DynamoDB.")

    missing = [i["ProductId"]["S"] for i in scan_all(ddb)
               if "DescriptionVector" not in i]
    if missing:
        print(f"\nWARNING: {len(missing)} items still have no vector.")
    else:
        print("All items have a DescriptionVector.")
        print("\nNext: step3_add_vector_index.py  (DIMENSIONS stays 1024)")
        print("Then search -- but embed queries with input_type='search_query'.")


if __name__ == "__main__":
    main()
