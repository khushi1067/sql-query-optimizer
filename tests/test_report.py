from optimizer.report import render_report


def test_report_escapes_sql_and_name():
    dataset = {"display_name":"<script>alert(1)</script>","rows":10,"columns":[("id","BIGINT")]}
    m = {"planning_ms": .2,"buffer_hits":4,"buffer_reads":0}
    r = [{"name":"Query A", "median_ms":10., "min_ms":9., "max_ms":11., "sql":"SELECT '<script>'", "metrics":m},
         {"name":"Query B", "median_ms":5., "min_ms":4., "max_ms":6., "sql":"SELECT 1", "metrics":m}]
    html = render_report(dataset,r)
    assert "2.00x" in html
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_report_includes_workspace_relationships_and_join_plans():
    dataset = {"display_name": "sales.csv", "rows": 2, "columns": [("customer_id", "BIGINT")]}
    workspace = {
        "sales": dataset, "customers": {**dataset, "display_name": "customers.csv"},
    }
    metrics = {"planning_ms": 0.2, "buffer_hits": 4, "buffer_reads": 0, "nodes": [
        {"node": "Hash Join", "join_type": "Left", "rows": 2},
    ]}
    results = [{"name": "Query A", "median_ms": 1, "min_ms": 1, "max_ms": 1,
                "sql": "SELECT * FROM sales LEFT JOIN customers USING (customer_id)", "metrics": metrics}]
    html = render_report(dataset, results, workspace=workspace)
    assert "sales.customer_id" in html and "customers.customer_id" in html
    assert "Hash Join (Left)" in html and "UNVERIFIED" in html
