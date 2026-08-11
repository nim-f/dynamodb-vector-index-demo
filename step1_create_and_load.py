"""
STEP 1 -- Create a plain DynamoDB table and load the parrot shop catalogue.

No embeddings. No Bedrock calls. No vector index. This is just an ordinary
DynamoDB table with 60 products in it -- the starting point most people are
actually in.

Runs in a few seconds and costs essentially nothing.

    python step1_create_and_load.py --region us-east-1
"""

import argparse
import json
import time

import boto3

TABLE = "ParrotShop"


def create_table(ddb):
    try:
        ddb.describe_table(TableName=TABLE)
        print(f"Table {TABLE} already exists, skipping creation.")
        return
    except ddb.exceptions.ResourceNotFoundException:
        pass

    ddb.create_table(
        TableName=TABLE,
        AttributeDefinitions=[
            {"AttributeName": "ProductId", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "ProductId", "KeyType": "HASH"},
        ],
        # On-demand from the start. Vector indexes require PAY_PER_REQUEST,
        # so choosing it now avoids a capacity mode switch in step 3.
        BillingMode="PAY_PER_REQUEST",
    )
    print(f"Creating table {TABLE}...")

    while True:
        status = ddb.describe_table(TableName=TABLE)["Table"]["TableStatus"]
        print(f"  TableStatus: {status}")
        if status == "ACTIVE":
            break
        time.sleep(3)


def to_item(p):
    """Plain item -- note there is no DescriptionVector attribute yet."""
    return {
        "ProductId":   {"S": p["ProductId"]},
        "Marketplace": {"S": p["Marketplace"]},
        "Category":    {"S": p["Category"]},
        "Name":        {"S": p["Name"]},
        "Description": {"S": p["Description"]},
        "Price":       {"N": str(p["Price"])},
        "Currency":    {"S": p["Currency"]},
        "PriceTier":   {"S": p["PriceTier"]},
    }


def load(ddb, products):
    """BatchWriteItem in chunks of 25, retrying UnprocessedItems."""
    written = 0

    for start in range(0, len(products), 25):
        chunk = products[start:start + 25]
        request = {TABLE: [{"PutRequest": {"Item": to_item(p)}} for p in chunk]}

        for attempt in range(1, 6):
            response = ddb.batch_write_item(RequestItems=request)
            unprocessed = response.get("UnprocessedItems", {})
            if not unprocessed.get(TABLE):
                break
            print(f"  retry {attempt}: {len(unprocessed[TABLE])} unprocessed")
            request = unprocessed
            time.sleep(2)
        else:
            raise RuntimeError("Items still unprocessed after 5 attempts")

        written += len(chunk)
        print(f"  {written}/{len(products)} written")

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--data", default="parrot-products.json")
    args = ap.parse_args()

    ddb = boto3.client("dynamodb", region_name=args.region)

    with open(args.data) as f:
        products = json.load(f)

    create_table(ddb)
    load(ddb, products)

    # DescribeTable's ItemCount refreshes roughly every six hours, so it will
    # read 0 right after a load. Scan with Select=COUNT to verify.
    count = ddb.scan(TableName=TABLE, Select="COUNT")["Count"]
    print(f"\nVerified item count (Scan): {count}")

    print("\nPlain table ready. No vectors, no index -- just data.")
    print("Next: step2_add_embeddings.py")


if __name__ == "__main__":
    main()
