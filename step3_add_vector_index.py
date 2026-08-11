"""
STEP 3 -- Add a vector index to the existing table and watch it backfill.

This is the interesting one. The table already has 60 items, each carrying a
DescriptionVector. UpdateTable adds the vector index, and DynamoDB backfills it
from the vectors already on those items -- at no charge.

You do not re-create the table. You do not rewrite the items.

    python step3_add_vector_index.py --region us-east-1
"""

import argparse
import time

import boto3

TABLE = "ParrotShop"
INDEX = "DescriptionIndex"
DIMENSIONS = 1024


def index_state(ddb):
    table = ddb.describe_table(TableName=TABLE)["Table"]
    for vi in table.get("VectorIndexes", []):
        if vi["IndexName"] == INDEX:
            return table["TableStatus"], vi["IndexStatus"], vi.get("Backfilling", False)
    return table["TableStatus"], None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    ddb = boto3.client("dynamodb", region_name=args.region)

    _, existing, _ = index_state(ddb)
    if existing:
        print(f"Vector index {INDEX} already exists (status: {existing}).")
    else:
        ddb.update_table(
            TableName=TABLE,
            VectorIndexUpdates=[
                {
                    "Create": {
                        "IndexName": INDEX,
                        "VectorAttribute": {"AttributeName": "DescriptionVector"},
                        "Dimensions": DIMENSIONS,
                        "DistanceFunction": "COSINE",
                        "SearchSchema": [
                            # Mandatory on every search once defined.
                            {"AttributeName": "Marketplace", "KeyType": "HASH"},
                            # Optional at search time.
                            {"AttributeName": "Category", "KeyType": "INLINE_FILTER"},
                            {"AttributeName": "PriceTier", "KeyType": "INLINE_FILTER"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    }
                }
            ],
        )
        print(f"Adding vector index {INDEX} to {TABLE}...")
        print("DynamoDB will backfill it from vectors already on the items,")
        print("at no additional charge.\n")

    started = time.time()
    while True:
        table_status, status, backfilling = index_state(ddb)
        elapsed = int(time.time() - started)
        print(f"  [{elapsed:>4}s] Table: {table_status} | "
              f"Index: {status} | Backfilling: {backfilling}")

        # Both conditions matter. Searching while Backfilling is true returns
        # incomplete results without raising an error.
        if table_status == "ACTIVE" and status == "ACTIVE" and not backfilling:
            break
        time.sleep(5)

    print(f"\nIndex ACTIVE and backfill complete after {int(time.time() - started)}s.")
    print("\nRemember: this configuration is now immutable. Dimensions,")
    print("distance function, projection, inline filters, and partition key")
    print("cannot be changed -- only replaced by a new index.")
    print("\nNext: search_parrot_products.py")


if __name__ == "__main__":
    main()
