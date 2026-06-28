#!/usr/bin/env python3
"""Generate deterministic medium-SQL training/eval JSONL.

The schema matches gretelai/synthetic_text_to_sql:
  {"sql_prompt": ..., "sql_context": ..., "sql": ...}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DOMAINS = [
    ("retail", "customers", "orders", "customer_id", "order_date", "amount", "region"),
    ("education", "students", "enrollments", "student_id", "enrolled_at", "score", "grade_level"),
    ("healthcare", "patients", "visits", "patient_id", "visit_date", "cost", "clinic"),
    ("logistics", "warehouses", "shipments", "warehouse_id", "shipped_at", "weight", "country"),
    ("finance", "clients", "transactions", "client_id", "transaction_date", "value", "segment"),
]

PRODUCTS = ["Fruit", "Electronics", "Books", "Furniture", "Apparel"]
REGIONS = ["Europe", "North America", "Asia-Pacific", "South West", "Canada"]
YEARS = [2019, 2020, 2021, 2022, 2023]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def base_schema(entity: str, fact: str, fk: str, date_col: str, metric: str, segment: str) -> str:
    return (
        f"CREATE TABLE {entity} (id INT, name TEXT, {segment} TEXT, created_at DATE);\n"
        f"CREATE TABLE {fact} (id INT, {fk} INT, {date_col} DATE, {metric} FLOAT, status TEXT, category TEXT);"
    )


def add_join_examples(rows: list[dict], rng: random.Random, n: int) -> None:
    for _ in range(n):
        _domain, entity, fact, fk, date_col, metric, segment = rng.choice(DOMAINS)
        status = rng.choice(["complete", "paid", "active", "delivered"])
        rows.append({
            "sql_prompt": f"List each {entity[:-1]} name with total {metric} for {status} {fact}.",
            "sql_context": base_schema(entity, fact, fk, date_col, metric, segment),
            "sql": (
                f"SELECT e.name, SUM(f.{metric}) AS total_{metric} "
                f"FROM {entity} e JOIN {fact} f ON e.id = f.{fk} "
                f"WHERE f.status = '{status}' GROUP BY e.name"
            ),
        })


def add_group_examples(rows: list[dict], rng: random.Random, n: int) -> None:
    for _ in range(n):
        _domain, entity, fact, fk, date_col, metric, segment = rng.choice(DOMAINS)
        year = rng.choice(YEARS)
        rows.append({
            "sql_prompt": f"Which {segment}s had average {metric} above 100 in {year}?",
            "sql_context": base_schema(entity, fact, fk, date_col, metric, segment),
            "sql": (
                f"SELECT e.{segment}, AVG(f.{metric}) AS avg_{metric} "
                f"FROM {entity} e JOIN {fact} f ON e.id = f.{fk} "
                f"WHERE EXTRACT(YEAR FROM f.{date_col}) = {year} "
                f"GROUP BY e.{segment} HAVING AVG(f.{metric}) > 100"
            ),
        })


def add_date_examples(rows: list[dict], rng: random.Random, n: int) -> None:
    for _ in range(n):
        _domain, entity, fact, fk, date_col, metric, segment = rng.choice(DOMAINS)
        year = rng.choice(YEARS)
        rows.append({
            "sql_prompt": f"How many {fact} happened in the first quarter of {year}?",
            "sql_context": base_schema(entity, fact, fk, date_col, metric, segment),
            "sql": (
                f"SELECT COUNT(*) FROM {fact} "
                f"WHERE {date_col} >= DATE '{year}-01-01' AND {date_col} < DATE '{year}-04-01'"
            ),
        })


def add_window_examples(rows: list[dict], rng: random.Random, n: int) -> None:
    for _ in range(n):
        _domain, entity, fact, fk, date_col, metric, segment = rng.choice(DOMAINS)
        rows.append({
            "sql_prompt": f"Show each {fact} row with the maximum {metric} for its {fk}.",
            "sql_context": base_schema(entity, fact, fk, date_col, metric, segment),
            "sql": (
                f"SELECT id, {fk}, {metric}, "
                f"MAX({metric}) OVER (PARTITION BY {fk}) AS max_{metric}_for_{fk} "
                f"FROM {fact}"
            ),
        })


def add_antijoin_examples(rows: list[dict], rng: random.Random, n: int) -> None:
    for _ in range(n):
        _domain, entity, fact, fk, date_col, metric, segment = rng.choice(DOMAINS)
        year = rng.choice(YEARS)
        rows.append({
            "sql_prompt": f"Which {entity} had no {fact} in {year}?",
            "sql_context": base_schema(entity, fact, fk, date_col, metric, segment),
            "sql": (
                f"SELECT e.name FROM {entity} e "
                f"WHERE NOT EXISTS (SELECT 1 FROM {fact} f WHERE f.{fk} = e.id "
                f"AND EXTRACT(YEAR FROM f.{date_col}) = {year})"
            ),
        })


def add_dml_examples(rows: list[dict], rng: random.Random, n: int) -> None:
    for i in range(n):
        table = rng.choice(["products", "employees", "tickets", "accounts"])
        rows.append({
            "sql_prompt": f"Update inactive {table} records to archived.",
            "sql_context": f"CREATE TABLE {table} (id INT, name TEXT, status TEXT, updated_at DATE);",
            "sql": f"UPDATE {table} SET status = 'archived' WHERE status = 'inactive'",
        })
        rows.append({
            "sql_prompt": f"Insert a new {table[:-1]} named Sample {i}.",
            "sql_context": f"CREATE TABLE {table} (id INT, name TEXT, status TEXT, updated_at DATE);",
            "sql": f"INSERT INTO {table} (name, status) VALUES ('Sample {i}', 'active')",
        })


def add_bigquery_ga4_examples(rows: list[dict], rng: random.Random, n: int) -> None:
    schema = """CREATE TABLE `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` (
  event_date STRING,
  event_timestamp INT64,
  event_name STRING,
  user_pseudo_id STRING,
  event_params ARRAY<STRUCT<
    key STRING,
    value STRUCT<string_value STRING, int_value INT64, float_value FLOAT64, double_value FLOAT64>
  >>,
  ecommerce STRUCT<total_item_quantity INT64, purchase_revenue_in_usd FLOAT64, transaction_id STRING>,
  items ARRAY<STRUCT<item_id STRING, item_name STRING, item_category STRING, price FLOAT64, quantity INT64>>
);"""
    for _ in range(n):
        start, end = rng.choice([
            ("20210101", "20210107"),
            ("20201201", "20201231"),
            ("20201125", "20201130"),
        ])
        rows.append({
            "sql_prompt": f"How many distinct users had positive engagement time between {start} and {end}?",
            "sql_context": schema,
            "sql": (
                "SELECT COUNT(DISTINCT user_pseudo_id) "
                "FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` "
                f"WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}' "
                "AND (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'engagement_time_msec') > 0"
            ),
        })
        rows.append({
            "sql_prompt": f"List item categories and total quantity purchased between {start} and {end}.",
            "sql_context": schema,
            "sql": (
                "SELECT item.item_category, SUM(item.quantity) AS total_quantity "
                "FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`, UNNEST(items) AS item "
                f"WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}' "
                "AND event_name = 'purchase' GROUP BY item.item_category"
            ),
        })


def build_dataset(seed: int, size: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    add_join_examples(rows, rng, size // 7)
    add_group_examples(rows, rng, size // 7)
    add_date_examples(rows, rng, size // 7)
    add_window_examples(rows, rng, size // 7)
    add_antijoin_examples(rows, rng, size // 7)
    add_dml_examples(rows, rng, max(1, size // 14))
    add_bigquery_ga4_examples(rows, rng, max(1, size // 14))
    while len(rows) < size:
        add_join_examples(rows, rng, 1)
        if len(rows) >= size:
            break
        add_group_examples(rows, rng, 1)
        if len(rows) >= size:
            break
        add_bigquery_ga4_examples(rows, rng, 1)
    rng.shuffle(rows)
    return rows[:size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", default="data/custom_sql_train.jsonl")
    parser.add_argument("--eval-out", default="data/custom_sql_eval.jsonl")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_rows = build_dataset(args.seed, args.train_size)
    eval_rows = build_dataset(args.seed + 10_000, args.eval_size)
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)
    print(f"wrote {len(train_rows)} train rows -> {args.train_out}")
    print(f"wrote {len(eval_rows)} eval rows -> {args.eval_out}")


if __name__ == "__main__":
    main()
