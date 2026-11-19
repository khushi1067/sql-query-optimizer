"""Read-only PostgreSQL analysis. AST checking is defense in depth, not a sandbox."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median
from pglast import parse_sql, ast
from pglast.stream import RawStream
from psycopg import errors, pq
from .config import SCHEMA, BENCHMARK_RUNS, MAX_QUERIES
from .data import DatasetError, MANAGED_TABLE_NAME, validate_table_name
from .db import read_connection


TableScope = str | Mapping[str, str]
JOIN_NODES = {"Hash Join", "Merge Join", "Nested Loop"}
PREVIEW_ROW_LIMIT = 100


class QueryError(ValueError):
    pass


@dataclass
class QueryPreview:
    columns: list[str]
    rows: list[tuple]
    truncated: bool


SAFE_FUNCTIONS = frozenset({
    "count", "sum", "avg", "min", "max", "coalesce", "nullif", "round",
    "abs", "lower", "upper", "length", "trim", "btrim", "ltrim", "rtrim",
    "substring", "substr", "date_trunc", "date_part", "to_char", "concat",
    "greatest", "least", "floor", "ceil", "ceiling", "mod", "power",
    "string_agg", "array_agg", "stddev", "stddev_pop", "stddev_samp",
    "variance", "var_pop", "var_samp", "extract", "date", "now",
    "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
    "lag", "lead", "first_value", "last_value", "nth_value",
})
FORBIDDEN_NODES = {"InsertStmt", "UpdateStmt", "DeleteStmt", "MergeStmt",
                   "CreateStmt", "CopyStmt", "CallStmt", "DoStmt", "AlterTableStmt",
                   "DropStmt", "ExecuteStmt", "VariableSetStmt", "TransactionStmt",
                   "LockStmt", "IntoClause", "SelectInto"}


def _table_scope(table: TableScope, friendly_name: str | None) -> dict[str, str]:
    if isinstance(table, str):
        table = {friendly_name if friendly_name is not None else f"{SCHEMA}.{table}": table}
    if not table:
        raise QueryError("Upload at least one table before running SQL.")
    for name, physical in table.items():
        if not MANAGED_TABLE_NAME.fullmatch(physical):
            raise QueryError("The workspace has an invalid internal table.")
        if name != f"{SCHEMA}.{physical}":
            try:
                validate_table_name(name)
            except DatasetError as exc:
                raise QueryError("The workspace has an invalid friendly table name.") from exc
    return dict(table)


def _resolve_query(query: str, table: TableScope, friendly_name: str | None) -> tuple[str, str, set[str]]:
    if not query or len(query) > 20_000:
        raise QueryError("SQL must be between 1 and 20,000 characters.")
    tables = _table_scope(table, friendly_name)
    has_friendly_names = any("." not in name for name in tables)
    try:
        statements = parse_sql(query)
    except Exception as exc:
        raise QueryError("SQL syntax is invalid.") from exc
    if len(statements) != 1 or not isinstance(statements[0].stmt, ast.SelectStmt):
        raise QueryError("Only one SELECT query is permitted.")

    root = statements[0].stmt
    referenced_tables = set()

    def visit(node, visible_ctes: frozenset[str]):
        if isinstance(node, (tuple, list)):
            for item in node:
                visit(item, visible_ctes)
            return
        if not isinstance(node, ast.Node):
            return
        if type(node).__name__ in FORBIDDEN_NODES:
            raise QueryError("Modification and administrative SQL are prohibited.")
        if isinstance(node, ast.ParamRef):
            raise QueryError("Use literal values instead of positional SQL parameters.")
        if isinstance(node, ast.SelectStmt):
            if node.intoClause or node.lockingClause:
                raise QueryError("SELECT INTO and locking statements are prohibited.")
            if node.withClause:
                clause = node.withClause
                if clause.recursive:
                    visible_ctes = visible_ctes.union(cte.ctename for cte in clause.ctes)
                for cte in clause.ctes:
                    visit(cte, visible_ctes)
                    visible_ctes = visible_ctes.union((cte.ctename,))
        if isinstance(node, ast.RangeVar):
            if node.catalogname is None and node.schemaname is None and node.relname in visible_ctes:
                return
            name = node.relname
            physical = None
            if node.catalogname is None and node.schemaname in (None, SCHEMA):
                physical = tables.get(name)
                if physical is None and node.schemaname == SCHEMA:
                    name = f"{SCHEMA}.{name}"
                    physical = tables.get(name)
            if physical is None:
                raise QueryError("Queries can reference only workspace tables and in-scope CTEs.")
            referenced_tables.add(name)
            if "." not in name:
                if node.alias is None:
                    node.alias = ast.Alias(aliasname=name)
                node.schemaname = SCHEMA
                node.relname = physical
        if has_friendly_names and isinstance(node, ast.ColumnRef) and len(node.fields) > 2:
            raise QueryError("Qualify columns with a table name or alias, such as customers.id or c.id.")
        if isinstance(node, ast.RangeTableSample):
            if len(node.method) != 1 or node.method[0].sval not in {"bernoulli", "system"}:
                raise QueryError("Only PostgreSQL's built-in sampling methods are permitted.")
        if isinstance(node, ast.FuncCall):
            parts = [p.sval for p in node.funcname]
            if len(parts) != 1 or parts[0].lower() not in SAFE_FUNCTIONS:
                raise QueryError(f"Function not permitted: {'.'.join(parts)}")
        for key in node.__slots__:
            if isinstance(node, ast.SelectStmt) and key == "withClause":
                continue
            visit(getattr(node, key, None), visible_ctes)

    try:
        visit(root, frozenset())
        if not referenced_tables:
            raise QueryError("Query must reference your uploaded table.")
        resolved = RawStream()(root)
    except RecursionError as exc:
        raise QueryError("SQL nesting is too deep.") from exc
    return query.strip().rstrip(";"), resolved, referenced_tables


def _resolve_sql(query: str, table: TableScope, friendly_name: str | None) -> tuple[str, str]:
    submitted, resolved, _ = _resolve_query(query, table, friendly_name)
    return submitted, resolved


def validate_generated_sql(query: str, tables: Mapping[str, str]) -> tuple[str, set[str]]:
    """Check PostgreSQL name binding with protocol Parse, without planning or running the query."""
    submitted, resolved, references = _resolve_query(query, tables, None)
    with read_connection() as conn:
        encoding = conn.info.encoding
        result = conn.pgconn.prepare(b"", resolved.encode(encoding))
        if result.status != pq.ExecStatus.COMMAND_OK:
            raise errors.error_from_result(result, encoding=encoding)
    return submitted, references


def validate_sql(query: str, table: TableScope, *, friendly_name: str | None = None) -> str:
    return _resolve_sql(query, table, friendly_name)[0]


def run_query_preview(query: str, tables: Mapping[str, str]) -> QueryPreview:
    _, resolved = _resolve_sql(query, tables, None)
    with read_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM ({resolved}) AS generated_result LIMIT {PREVIEW_ROW_LIMIT + 1}")
            columns = [column.name for column in cursor.description]
            rows = cursor.fetchall()
    return QueryPreview(columns, rows[:PREVIEW_ROW_LIMIT], len(rows) > PREVIEW_ROW_LIMIT)


def plan_metrics(plan: dict, *, relation_names: Mapping[str, str] | None = None) -> dict:
    root = plan["Plan"]
    nodes = []
    def scan(node, depth=0):
        relation = node.get("Relation Name", "")
        if relation_names is not None:
            relation = relation_names.get(relation, relation)
        nodes.append({"depth": depth, "node": node.get("Node Type", "Unknown"),
                      "relation": relation,
                      "rows": node.get("Actual Rows", 0),
                      "estimated": node.get("Plan Rows", 0),
                      "time": node.get("Actual Total Time", 0),
                      "cost": node.get("Total Cost", 0)})
        if node.get("Node Type") in JOIN_NODES:
            nodes[-1]["join_type"] = node.get("Join Type", "")
        for sub in node.get("Plans", []):
            scan(sub, depth + 1)
    scan(root)
    return {"execution_ms": float(plan.get("Execution Time", 0)),
            "planning_ms": float(plan.get("Planning Time", 0)),
            "buffer_hits": int(root.get("Shared Hit Blocks", 0)),
            "buffer_reads": int(root.get("Shared Read Blocks", 0)),
            "rows": int(root.get("Actual Rows", 0)),
            "nodes": nodes,
            "join_operations": [n for n in nodes if n["node"] in JOIN_NODES],
            "plan": plan}


def explain(sql_text: str, table: TableScope, *, friendly_name: str | None = None) -> dict:
    _, resolved = _resolve_sql(sql_text, table, friendly_name)
    return _explain_resolved(resolved, _table_scope(table, friendly_name))


def _explain_resolved(resolved: str, tables: Mapping[str, str]) -> dict:
    # SQL text cannot become administrative SQL: validated AST + restricted role + READ ONLY transaction.
    with read_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + resolved)
            value = cursor.fetchone()[0]
            return plan_metrics(value[0], relation_names={physical: name for name, physical in tables.items() if "." not in name})


def compare(queries: list[str], table: TableScope, runs: int = BENCHMARK_RUNS, *,
            friendly_name: str | None = None) -> list[dict]:
    if not 2 <= len(queries) <= MAX_QUERIES:
        raise QueryError(f"Provide 2 to {MAX_QUERIES} queries.")
    tables = _table_scope(table, friendly_name)
    checked = [_resolve_sql(q, tables, None) for q in queries]
    if runs not in (1, 5):
        raise QueryError("Benchmarks must use 1 or 5 runs.")
    results = [[] for _ in checked]
    for round_index in range(runs):
        indexes = list(range(len(checked)))
        if round_index % 2:
            indexes.reverse()
        for i in indexes:
            results[i].append(_explain_resolved(checked[i][1], tables))
    output = []
    for i, repeats in enumerate(results):
        times = [r["execution_ms"] for r in repeats]
        representative = min(repeats, key=lambda r: abs(r["execution_ms"] - median(times)))
        output.append({"name": f"Query {chr(65+i)}", "sql": checked[i][0],
                       "median_ms": median(times), "min_ms": min(times),
                       "max_ms": max(times), "runs": times, "metrics": representative})
    return output


def insights(metrics: dict) -> list[str]:
    nodes = [n["node"] for n in metrics["nodes"]]
    findings = []
    if "Seq Scan" in nodes:
        findings.append("PostgreSQL used a sequential scan. This can be optimal for small tables or broad filters; it is not automatically a problem.")
    if any("Index" in n for n in nodes):
        findings.append("The execution plan includes index access, which can reduce work for selective predicates.")
    if any("Sort" in n for n in nodes):
        findings.append("The plan includes sorting. Check whether ORDER BY is required and how many rows are sorted.")
    joins = sorted(JOIN_NODES.intersection(nodes))
    if joins:
        findings.append("Join operations: " + ", ".join(joins) + ". Compare intermediate row counts.")
    if metrics["buffer_reads"]:
        findings.append("Shared buffer reads occurred. Repeated benchmark timings can vary with cache state.")
    if not findings:
        findings.append("Inspect the plan nodes and row counts to understand PostgreSQL's chosen strategy.")
    return findings
