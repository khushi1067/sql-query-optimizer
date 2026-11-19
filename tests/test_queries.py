import pytest
from optimizer.query import validate_sql, QueryError, plan_metrics, insights

T = "ds_" + "a" * 32


def test_safe_select():
    q = f"SELECT category, SUM(amount) FROM optimizer_data.{T} GROUP BY category"
    assert validate_sql(q,T)==q


@pytest.mark.parametrize("sql", [
    "DELETE FROM optimizer_data.{t}",
    "SELECT * FROM optimizer_data.{t}; DROP TABLE optimizer_data.{t}",
    "SELECT * FROM public.customers",
    "SELECT pg_sleep(1) FROM optimizer_data.{t}",
    "WITH changed AS (DELETE FROM optimizer_data.{t} RETURNING *) SELECT * FROM changed",
    "SELECT * INTO TEMP x FROM optimizer_data.{t}",
    "SELECT * FROM optimizer_data.{t} FOR UPDATE",
])
def test_unsafe_sql(sql):
    with pytest.raises(QueryError):
        validate_sql(sql.format(t=T),T)


def test_plan_metrics_nested():
    plan = {"Execution Time": 12.5, "Planning Time": .4,
        "Plan": {"Node Type": "Aggregate", "Actual Rows": 1, "Shared Hit Blocks": 10,
                 "Plans": [{"Node Type": "Seq Scan", "Actual Rows": 100, "Plan Rows": 99}]}}
    value = plan_metrics(plan)
    assert value["execution_ms"] == 12.5
    assert len(value["nodes"]) == 2
    assert "sequential scan" in insights(value)[0]


@pytest.mark.parametrize("statement", [
    "SELECT customers.id FROM customers",
    'SELECT "customers".id FROM "optimizer_data"."customers"',
    "SELECT c.id FROM customers AS c",
    "SELECT a.id FROM customers AS a JOIN customers AS b ON a.id = b.id",
    "SELECT q.id FROM (SELECT customers.id FROM customers) AS q",
    "WITH source AS (SELECT id FROM customers), ids AS (SELECT id FROM source) SELECT * FROM ids",
    "WITH customers AS (SELECT id FROM optimizer_data.customers) SELECT * FROM customers",
    "WITH RECURSIVE ids AS (SELECT id FROM customers UNION ALL SELECT id FROM ids WHERE false) SELECT * FROM ids",
])
def test_resolves_friendly_table(statement):
    from pglast import ast, parse_sql

    from optimizer.query import _resolve_sql

    submitted, executable = _resolve_sql(statement, T, "customers")
    assert submitted == statement
    assert f"optimizer_data.{T}" in executable
    assert isinstance(parse_sql(executable)[0].stmt, ast.SelectStmt)


def test_rewrites_relations_without_rewriting_literals_or_aliases():
    from optimizer.query import _resolve_sql

    statement = "SELECT customers.id, 'customers' FROM customers WHERE customers.id = 1"
    _, executable = _resolve_sql(statement, T, "customers")
    assert executable == (
        f"SELECT customers.id, 'customers' FROM optimizer_data.{T} AS customers WHERE customers.id = 1"
    )


@pytest.mark.parametrize("statement", [
    "SELECT * FROM public.customers",
    "SELECT * FROM database_name.optimizer_data.customers",
    'SELECT * FROM "Customers"',
    "SELECT * FROM optimizer_data." + T,
    "SELECT * FROM customers JOIN optimizer_data.ds_" + "b" * 32 + " AS other ON true",
    "SELECT * FROM customers JOIN private_table ON true",
    "WITH a AS (SELECT * FROM later), later AS (SELECT * FROM customers) SELECT * FROM a",
    "SELECT * FROM customers WHERE EXISTS (SELECT 1 FROM private_table) "
    "AND EXISTS (WITH private_table AS (SELECT 1) SELECT 1 FROM private_table)",
    "WITH customers AS (SELECT 1) SELECT * FROM customers",
    "SELECT optimizer_data.customers.id FROM optimizer_data.customers",
    "SELECT set_config('search_path', 'public', true) FROM customers",
    "SELECT * FROM customers; SET search_path = public",
])
def test_friendly_mode_rejects_unrelated_references(statement):
    with pytest.raises(QueryError):
        validate_sql(statement, T, friendly_name="customers")


def test_only_resolved_sql_reaches_explain(monkeypatch):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    from optimizer import query

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ([{
        "Plan": {"Node Type": "Seq Scan", "Relation Name": T}, "Execution Time": 1.5,
    }],)

    @contextmanager
    def connect():
        yield connection

    monkeypatch.setattr(query, "read_connection", connect)
    statements = ["SELECT * FROM customers", "SELECT c.id FROM customers AS c"]
    results = query.compare(statements, T, runs=5, friendly_name="customers")
    assert [item["sql"] for item in results] == statements
    assert len(cursor.execute.call_args_list) == 10
    assert {
        call.args[0] for call in cursor.execute.call_args_list
    } == {
        f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT * FROM optimizer_data.{T} AS customers",
        f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT c.id FROM optimizer_data.{T} AS c",
    }
    assert results[0]["metrics"]["nodes"][0]["relation"] == "customers"
    assert results[0]["metrics"]["plan"]["Plan"]["Relation Name"] == T
    assert query.explain(statements[0], T, friendly_name="customers")["execution_ms"] == 1.5
    cursor.execute.reset_mock()
    with pytest.raises(QueryError):
        query.compare([statements[0], "SELECT * FROM public.customers"], T, friendly_name="customers")
    cursor.execute.assert_not_called()


WORKSPACE_TABLES = {"sales": "ds_" + "a" * 32, "customers": "ds_" + "b" * 32, "categories": "ds_" + "c" * 32}


@pytest.mark.parametrize("join", ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN", "CROSS JOIN"])
def test_workspace_joins(join):
    from optimizer.query import _resolve_sql

    condition = "" if join == "CROSS JOIN" else " ON sales.customer_id = c.customer_id"
    query = f"SELECT sales.* FROM sales {join} optimizer_data.customers AS c{condition}"
    submitted, executable = _resolve_sql(query, WORKSPACE_TABLES, None)
    assert submitted == query
    assert f"optimizer_data.{WORKSPACE_TABLES['sales']} AS sales" in executable
    assert f"optimizer_data.{WORKSPACE_TABLES['customers']} AS c" in executable


def test_three_tables_window_cte_shadowing_and_aggregation():
    from optimizer.query import _resolve_sql

    query = """
        WITH sales AS (SELECT * FROM optimizer_data.sales)
        SELECT c.customer_id, sum(s.amount), rank() OVER (ORDER BY sum(s.amount) DESC)
        FROM sales AS s
        JOIN customers AS c USING (customer_id)
        JOIN (SELECT category_id FROM categories) AS cat USING (category_id)
        GROUP BY c.customer_id HAVING sum(s.amount) > 0 ORDER BY c.customer_id
    """
    _, executable = _resolve_sql(query, WORKSPACE_TABLES, None)
    assert "FROM sales AS s" in executable
    assert all(physical in executable for physical in WORKSPACE_TABLES.values())
    assert "rank() OVER" in executable


@pytest.mark.parametrize("query", [
    "SELECT * FROM sales JOIN public.customers ON true",
    "SELECT * FROM sales JOIN optimizer_data." + WORKSPACE_TABLES["customers"] + " ON true",
    "SELECT * FROM sales JOIN optimizer_data.ds_" + "f" * 32 + " ON true",
    "SELECT pg_sleep(1) FROM sales JOIN customers ON true",
    "SELECT * FROM sales TABLESAMPLE foreign_handler(1)",
    "WITH c AS (SELECT * FROM external) SELECT * FROM c JOIN sales ON true",
])
def test_workspace_rejects_external_relations_and_unsafe_functions(query):
    with pytest.raises(QueryError):
        validate_sql(query, WORKSPACE_TABLES)


def test_join_operation_details():
    plan = {"Plan": {"Node Type": "Hash Join", "Join Type": "Inner", "Plans": [
        {"Node Type": "Merge Join", "Join Type": "Left"},
        {"Node Type": "Nested Loop", "Join Type": "Full"},
    ]}}
    metrics = plan_metrics(plan)
    assert [(node["node"], node["join_type"]) for node in metrics["join_operations"]] == [
        ("Hash Join", "Inner"), ("Merge Join", "Left"), ("Nested Loop", "Full"),
    ]


def test_four_table_join_and_unauthorized_references():
    from optimizer.query import _resolve_sql

    scope = {
        name: "ds_" + character * 32
        for name, character in zip(("customers", "orders", "products", "order_items"), "abcd")
    }
    sql = """
        WITH selected_orders AS (SELECT * FROM orders)
        SELECT c.region, COUNT(*)
        FROM customers AS c JOIN selected_orders AS o USING (customer_id)
        JOIN (SELECT * FROM order_items) AS i USING (order_id)
        JOIN products AS p USING (product_id) GROUP BY c.region
    """.strip()
    submitted, executable = _resolve_sql(sql, scope, None)
    assert submitted == sql
    assert all(f"optimizer_data.{physical}" in executable for physical in scope.values())
    assert "JOIN selected_orders AS o" in executable
    for other in ("public.customers", "optimizer_data.ds_" + "e" * 32):
        with pytest.raises(QueryError):
            validate_sql(f"SELECT * FROM customers c JOIN {other} ON true", scope)
    assert validate_sql("SELECT * FROM customers", {"customers": scope["customers"]})


def test_generated_query_preview_revalidates_and_limits_output(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from optimizer import query

    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.description = [SimpleNamespace(name="id")]
    cursor.fetchall.return_value = [(index,) for index in range(query.PREVIEW_ROW_LIMIT + 1)]

    @contextmanager
    def read_connection():
        yield conn

    monkeypatch.setattr(query, "read_connection", read_connection)
    scope = {"customers": T}
    with pytest.raises(QueryError):
        query.run_query_preview("DELETE FROM customers", scope)
    with pytest.raises(QueryError):
        query.run_query_preview("SELECT * FROM public.customers", scope)
    cursor.execute.assert_not_called()

    preview = query.run_query_preview("SELECT id % 2 FROM customers", scope)
    assert preview.columns == ["id"] and preview.truncated
    assert len(preview.rows) == query.PREVIEW_ROW_LIMIT
    cursor.execute.assert_called_once()
    executed = cursor.execute.call_args.args[0]
    assert f"optimizer_data.{T} AS customers" in executed
    assert executed.endswith("LIMIT 101") and "%" in executed
