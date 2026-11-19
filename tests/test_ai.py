import json
from unittest.mock import Mock, call

import pytest
from google.genai import errors, types

from optimizer import ai
from optimizer.report import render_report


TABLE = "ds_" + "a" * 32
QUALIFIED = f"optimizer_data.{TABLE}"
FRIENDLY = "customers"
DATASET = {
    "table": TABLE, "qualified": QUALIFIED, "display_name": "fixture.csv",
    "rows": 2, "friendly_name": FRIENDLY, "columns": [("id", "BIGINT")],
}
RESULTS = [
    {
        "name": f"Query {name}", "sql": f"SELECT * FROM {FRIENDLY} LIMIT 1",
        "median_ms": ms, "min_ms": ms, "max_ms": ms,
        "metrics": {"planning_ms": 0.1, "buffer_hits": 0, "buffer_reads": 0, "nodes": []},
    }
    for name, ms in (("A", 2), ("B", 3))
]
SUGGESTION = {
    "optimized_sql": f"SELECT id FROM {FRIENDLY} LIMIT 1;",
    "explanation": "Select the only column explicitly.",
    "expected_mechanism": "The execution plan may remain the same.",
    "cautions": "Verify output before use.",
    "no_safe_optimization": False,
}


def response(text):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=text)]))]
    )


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-fixture")
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=client)
    monkeypatch.setattr(ai.genai, "Client", factory)
    delay = Mock()
    monkeypatch.setattr(ai, "sleep", delay)
    return factory, client.models.generate_content, delay


def suggest():
    return ai.get_suggestion(TABLE, DATASET["columns"], RESULTS[0], RESULTS, friendly_name=FRIENDLY)


@pytest.mark.parametrize("fenced", [False, True])
def test_suggestion_and_schema(gemini, fenced):
    factory, generate, delay = gemini
    body = json.dumps(SUGGESTION)
    generate.return_value = response(f"```json\n{body}\n```" if fenced else body)
    assert suggest().optimized_sql == SUGGESTION["optimized_sql"].rstrip(";")
    request = generate.call_args.kwargs
    assert request["model"] == "gemini-fixture"
    assert "config" not in request
    assert all(f'"{field}"' in request["contents"] for field in SUGGESTION)
    http_options = factory.call_args.kwargs["http_options"]
    assert http_options.timeout >= 10_000
    assert http_options.retry_options.attempts == 1
    assert "test-key" not in request["contents"]
    assert TABLE not in request["contents"]
    assert '"authorized_table": "customers"' in request["contents"]
    assert '"optional_sample_rows": "not supplied"' in request["contents"]
    delay.assert_not_called()


@pytest.mark.parametrize(("body", "category"), [
    ("{invalid fixture-secret", "malformed_json"),
    (json.dumps({"optimized_sql": "fixture-secret"}), "invalid_fields"),
    (json.dumps({**SUGGESTION, "no_safe_optimization": "true"}), "invalid_fields"),
    (json.dumps({**SUGGESTION, "explanation": " "}), "invalid_fields"),
    (None, "empty_response"),
])
def test_response_failures(gemini, caplog, body, category):
    _, generate, delay = gemini
    generate.return_value = response(body)
    with pytest.raises(ai.AdvisorError):
        suggest()
    assert f"stage=response_parsing category={category}" in caplog.text
    assert "fixture-secret" not in caplog.text
    generate.assert_called_once()
    delay.assert_not_called()


@pytest.mark.parametrize("sql", ["DELETE FROM " + FRIENDLY, "SELECT * FROM another_table", "SELECT * FROM " + QUALIFIED])
def test_unsafe_sql(gemini, caplog, sql):
    _, generate, delay = gemini
    generate.return_value = response(json.dumps({**SUGGESTION, "optimized_sql": sql}))
    with pytest.raises(ai.AdvisorError, match="allowed SELECT"):
        suggest()
    assert "stage=validation category=invalid_sql" in caplog.text
    generate.assert_called_once()
    delay.assert_not_called()


def test_no_improvement(gemini):
    _, generate, _ = gemini
    generate.return_value = response(json.dumps({
        **SUGGESTION, "optimized_sql": "", "no_safe_optimization": True,
    }))
    proposal = suggest()
    assert proposal.no_safe_optimization and not proposal.optimized_sql
    assert "No safe optimization recommended" in render_report(DATASET, RESULTS, proposal)


@pytest.mark.parametrize(("code", "category"), [(429, "rate_limit"), (503, "unavailable")])
def test_transient_retries(gemini, caplog, code, category):
    _, generate, delay = gemini
    generate.side_effect = errors.APIError(code, {"error": {"message": "fixture-secret"}})
    with pytest.raises(ai.AdvisorError):
        suggest()
    assert generate.call_count == 3
    assert delay.call_args_list == [call(0.5), call(1.0)]
    assert f"category={category}" in caplog.text and f"error_code={code}" in caplog.text
    assert "fixture-secret" not in caplog.text


def test_transient_recovers(gemini):
    _, generate, delay = gemini
    generate.side_effect = [errors.APIError(503, {}), response(json.dumps(SUGGESTION))]
    assert suggest().optimized_sql
    delay.assert_called_once_with(0.5)


@pytest.mark.parametrize(("code", "category"), [(400, "invalid_request"), (401, "authentication"), (404, "model_unavailable")])
def test_permanent_api_error(gemini, caplog, code, category):
    _, generate, delay = gemini
    generate.side_effect = errors.APIError(code, {"error": {"message": "fixture-secret"}})
    with pytest.raises(ai.AdvisorError):
        suggest()
    assert f"category={category}" in caplog.text and f"error_code={code}" in caplog.text
    assert "fixture-secret" not in caplog.text
    generate.assert_called_once()
    delay.assert_not_called()


@pytest.mark.parametrize("setting", ["GEMINI_API_KEY", "GEMINI_MODEL"])
def test_missing_configuration(gemini, monkeypatch, setting):
    factory, _, _ = gemini
    monkeypatch.delenv(setting)
    with pytest.raises(ai.AdvisorError, match=setting):
        suggest()
    factory.assert_not_called()


def advisor_dashboard(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from optimizer import db

    monkeypatch.setattr(db, "workspace_indexes", lambda tables: {name: [] for name in tables})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["dataset"] = DATASET
    app.session_state["queries"] = [result["sql"] for result in RESULTS]
    app.session_state["results"] = RESULTS
    app.session_state["ai_consent"] = True
    return app.run()


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def test_dashboard_suggestion_benchmark_and_failure_preserve_state(gemini, monkeypatch):
    from optimizer import query

    _, generate, _ = gemini
    generate.return_value = response(json.dumps(SUGGESTION))
    benchmark = Mock(return_value=RESULTS)
    monkeypatch.setattr(query, "compare", benchmark)
    app = advisor_dashboard(monkeypatch)

    click(app, "Ask AI to optimize")
    proposal = app.session_state["ai_suggestion"]
    assert any(block.value == proposal.optimized_sql for block in app.code)
    benchmark.assert_not_called()

    click(app, "Test AI Query")
    benchmark.assert_called_once_with([RESULTS[0]["sql"], proposal.optimized_sql], {FRIENDLY: TABLE}, runs=5)
    assert [metric.label for metric in app.metric][-2:] == ["Retested original", "AI candidate"]

    generate.return_value = response("broken response")
    click(app, "Ask AI to optimize")
    assert "invalid or incomplete" in app.error[0].value
    assert app.session_state["results"] == RESULTS
    assert app.session_state["ai_suggestion"] == proposal
    assert app.session_state["ai_result"] == RESULTS[1]
    assert len(app.get("download_button")) == 1
    report = render_report(DATASET, RESULTS, proposal, RESULTS[1])
    assert "<code>customers</code>" in report and TABLE not in report
    assert "AI optimization proposal" in report and "Measured AI candidate median" in report


def test_dashboard_no_improvement(gemini, monkeypatch):
    _, generate, _ = gemini
    generate.return_value = response(json.dumps({
        **SUGGESTION, "optimized_sql": "", "no_safe_optimization": True,
    }))
    app = advisor_dashboard(monkeypatch)
    click(app, "Ask AI to optimize")
    assert any("No safe optimization recommended" in item.value for item in app.info)
    assert all(button.label != "Test AI Query" for button in app.button)
    assert len(app.get("download_button")) == 1


def test_sdk_sends_plain_text_over_http(monkeypatch):
    import httpx2

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-fixture")
    client = ai.genai.Client
    seen = []

    def reply(request):
        seen.append(request)
        assert int(request.headers["X-Server-Timeout"]) >= 10
        body = json.loads(request.content)
        assert "generationConfig" not in body
        prompt = body["contents"][0]["parts"][0]["text"]
        assert "Return only a JSON object as text" in prompt
        return httpx2.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": json.dumps(SUGGESTION)}]}}],
        })

    def mock_transport(**kwargs):
        kwargs["http_options"].client_args = {"transport": httpx2.MockTransport(reply)}
        return client(**kwargs)

    monkeypatch.setattr(ai.genai, "Client", mock_transport)
    assert suggest().optimized_sql == SUGGESTION["optimized_sql"].rstrip(";")
    assert len(seen) == 1
    assert seen[0].url.path.endswith("/models/gemini-fixture:generateContent")


def test_key_error_with_http_400(gemini, caplog):
    _, generate, delay = gemini
    generate.side_effect = errors.APIError(400, {
        "error": {"details": [{"reason": "API_KEY_INVALID", "metadata": {"key": "fixture-secret"}}]},
    })
    with pytest.raises(ai.AdvisorError, match="rejected the configured key"):
        suggest()
    assert "category=authentication" in caplog.text
    assert "fixture-secret" not in caplog.text
    generate.assert_called_once()
    delay.assert_not_called()


@pytest.mark.parametrize(("status", "details", "message", "category"), [
    ("INVALID_ARGUMENT", [], "fixture-secret", "invalid_argument"),
    ("INVALID_ARGUMENT", [], "Manually set deadline 8s is too short. Minimum allowed deadline is 10s. fixture-secret", "deadline_too_short"),
    ("INVALID_ARGUMENT", [{"fieldViolations": [{"description": "fixture-secret"}]}], "", "request_fields"),
    ("INVALID_ARGUMENT", [], "Input token count exceeds limit. fixture-secret", "input_size"),
    ("FAILED_PRECONDITION", [], "User location is not supported. fixture-secret", "region"),
    ("FAILED_PRECONDITION", [{"reason": "BILLING_DISABLED"}], "fixture-secret", "billing"),
])
def test_provider_diagnostics_are_sanitized(gemini, caplog, status, details, message, category):
    _, generate, delay = gemini
    generate.side_effect = errors.APIError(400, {
        "error": {"status": status, "message": message, "details": details},
    })
    with pytest.raises(ai.AdvisorError) as failure:
        suggest()
    assert f"category={category}" in caplog.text
    assert "fixture-secret" not in caplog.text
    assert "fixture-secret" not in str(failure.value)
    generate.assert_called_once()
    delay.assert_not_called()


def test_gemini_workspace_join_uses_all_schemas_and_validates_query(gemini):
    _, generate, _ = gemini
    sales = "ds_" + "b" * 32
    workspace = {
        FRIENDLY: {**DATASET, "columns": [("customer_id", "BIGINT")]},
        "sales": {"table": sales, "columns": [("customer_id", "BIGINT"), ("amount", "BIGINT")], "rows": 42},
    }
    query = "SELECT sum(s.amount) FROM sales s JOIN customers c USING (customer_id)"
    generate.return_value = response(json.dumps({**SUGGESTION, "optimized_sql": query}))
    indexes = {"sales": [{"method": "btree", "key_columns": ["customer_id"]}], FRIENDLY: []}
    assert ai.get_suggestion(workspace, None, RESULTS[0], RESULTS, indexes=indexes).optimized_sql == query
    prompt = generate.call_args.kwargs["contents"]
    assert all(value in prompt for value in ("authorized_tables", "suggested_join_keys", "customer_id", "btree", '"rows": 42'))
    assert '"optional_sample_rows": "not supplied"' in prompt
    assert TABLE not in prompt and sales not in prompt
    generate.return_value = response(json.dumps({
        **SUGGESTION, "optimized_sql": f"SELECT * FROM sales JOIN optimizer_data.{TABLE} ON true",
    }))
    with pytest.raises(ai.AdvisorError, match="allowed SELECT"):
        ai.get_suggestion(workspace, None, RESULTS[0], RESULTS)


GENERATED = {
    "generated_sql": "SELECT c.id FROM customers AS c",
    "explanation": "Lists customer IDs.",
    "tables_used": ["customers"],
    "assumptions": [],
    "warnings": [],
}


@pytest.fixture
def parse_only(monkeypatch):
    from contextlib import contextmanager
    from psycopg import pq
    from optimizer import query

    conn = Mock()
    conn.info.encoding = "utf-8"
    conn.pgconn.prepare.return_value.status = pq.ExecStatus.COMMAND_OK

    @contextmanager
    def read_connection():
        yield conn

    monkeypatch.setattr(query, "read_connection", read_connection)
    return conn


def test_natural_language_single_table(gemini, parse_only):
    _, generate, _ = gemini
    generate.return_value = response(json.dumps(GENERATED))
    generated = ai.generate_sql("List customer IDs", {FRIENDLY: DATASET})
    assert generated.generated_sql == GENERATED["generated_sql"]
    args = generate.call_args.kwargs
    assert args["config"].response_mime_type == "application/json"
    assert args["config"].response_schema is None
    assert args["config"].response_json_schema is None
    assert "List customer IDs" in args["contents"] and TABLE not in args["contents"]
    parse_only.pgconn.prepare.assert_called_once_with(
        b"", f"SELECT c.id FROM optimizer_data.{TABLE} AS c".encode(),
    )
    parse_only.execute.assert_not_called()


def test_natural_language_join(gemini, parse_only):
    _, generate, _ = gemini
    orders = "ds_" + "b" * 32
    sql = "WITH o AS (SELECT id FROM orders) SELECT c.id FROM customers c LEFT JOIN o USING (id)"
    generate.return_value = response(json.dumps({**GENERATED, "generated_sql": sql, "tables_used": ["orders", "customers"]}))
    workspace = {FRIENDLY: DATASET, "orders": {**DATASET, "table": orders}}
    generated = ai.generate_sql("Find buyers and their orders", workspace)
    assert generated.tables_used == ["customers", "orders"]
    assert b"optimizer_data." + orders.encode() in parse_only.pgconn.prepare.call_args.args[1]
    assert '"suggested_join_keys"' in generate.call_args.kwargs["contents"]


@pytest.mark.parametrize("statement", [
    "SELECT * FROM invented",
    "DELETE FROM customers",
    "SELECT pg_sleep(1) FROM customers",
    "SELECT * FROM customers; COPY customers TO STDOUT",
])
def test_natural_language_rejects_unsafe_sql_and_invented_tables(gemini, parse_only, statement):
    _, generate, _ = gemini
    generate.return_value = response(json.dumps({**GENERATED, "generated_sql": statement}))
    with pytest.raises(ai.AdvisorError, match="allowed SELECT"):
        ai.generate_sql("List customers", {FRIENDLY: DATASET})
    parse_only.pgconn.prepare.assert_not_called()


def test_natural_language_rejects_invented_column(gemini, parse_only, monkeypatch):
    from psycopg import errors, pq
    from optimizer import query

    _, generate, _ = gemini
    generate.return_value = response(json.dumps({**GENERATED, "generated_sql": "SELECT invented FROM customers"}))
    parse_only.pgconn.prepare.return_value.status = pq.ExecStatus.FATAL_ERROR
    monkeypatch.setattr(query.errors, "error_from_result", lambda *_args, **_kwargs: errors.UndefinedColumn("private"))
    with pytest.raises(ai.AdvisorError, match="column that does not exist"):
        ai.generate_sql("List customers", {FRIENDLY: DATASET})
    parse_only.execute.assert_not_called()


@pytest.mark.parametrize("body", ["", "{broken", '{"generated_sql":"SELECT 1"}'])
def test_natural_language_invalid_response(gemini, parse_only, body):
    _, generate, _ = gemini
    generate.return_value = response(body)
    with pytest.raises(ai.AdvisorError):
        ai.generate_sql("List customers", {FRIENDLY: DATASET})
    parse_only.pgconn.prepare.assert_not_called()


def test_natural_language_clarification_and_missing_key(gemini, parse_only, monkeypatch):
    _, generate, _ = gemini
    generate.return_value = response(json.dumps(
        {**GENERATED, "generated_sql": "", "tables_used": [], "warnings": ["Which region?"]},
    ))
    assert ai.generate_sql("Show results there", {FRIENDLY: DATASET}).warnings == ["Which region?"]
    parse_only.pgconn.prepare.assert_not_called()
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ai.AdvisorError, match="GEMINI_API_KEY"):
        ai.generate_sql("List customers", {FRIENDLY: DATASET})
    generate.assert_called_once()


def test_generator_copies_and_runs_independently(gemini, parse_only, monkeypatch):
    from optimizer import query

    _, generate, _ = gemini
    generate.return_value = response(json.dumps(GENERATED))
    run = Mock(return_value=query.QueryPreview(["id"], [(1,)], False))
    monkeypatch.setattr(query, "run_query_preview", run)
    app = advisor_dashboard(monkeypatch)
    original_a = "SELECT id FROM customers LIMIT 23"
    original_b = "SELECT id FROM customers LIMIT 24"
    app.text_area(key="editor_0_0").set_value(original_a)
    app.text_area(key="editor_0_1").set_value(original_b)
    app.text_input(key="generator_question").set_value("List customers").run()

    click(app, "Generate SQL")
    assert app.text_area(key="generator_sql").value == GENERATED["generated_sql"]
    assert app.session_state["generator_explanation"] == GENERATED["explanation"]
    run.assert_not_called()
    assert not any(button.label.startswith("Use in Query") for button in app.button)

    edited = "SELECT '</script><script>bad()</script>' FROM customers"
    app.text_area(key="generator_sql").set_value(edited).run()
    copy_html = app.get("iframe")[0].proto.srcdoc
    assert "Copy SQL" in copy_html and "navigator.clipboard.writeText(sql)" in copy_html
    assert json.loads(copy_html.split("const sql = ", 1)[1].split(";\nconst status", 1)[0]) == edited
    assert "<script>bad()</script>" not in copy_html
    run.assert_not_called()

    click(app, "Run Generated Query")
    run.assert_called_once_with(edited, {FRIENDLY: TABLE})
    assert app.session_state["generator_results"][1].rows == [(1,)]
    assert any(item.value == "Generated query results" for item in app.subheader)
    assert app.text_area(key="editor_0_0").value == original_a
    assert app.text_area(key="editor_0_1").value == original_b
    assert app.session_state["results"] == RESULTS
    assert len(app.get("download_button")) == 1
    generate.assert_called_once()

    generate.return_value = response(json.dumps({**GENERATED, "generated_sql": "SELECT * FROM customers"}))
    app.text_input(key="generator_question").set_value("List all columns").run()
    click(app, "Generate SQL")
    assert app.text_area(key="generator_sql").value == "SELECT * FROM customers"
    assert "generator_results" not in app.session_state
    assert app.session_state["results"] == RESULTS
    assert app.text_area(key="editor_0_0").value == original_a
    assert app.text_area(key="editor_0_1").value == original_b
    run.assert_called_once()


def test_generator_error_stays_separate(gemini, parse_only, monkeypatch):
    _, generate, _ = gemini
    generate.return_value = response("malformed")
    app = advisor_dashboard(monkeypatch)
    app.text_input(key="generator_question").set_value("List customers").run()
    click(app, "Generate SQL")
    assert "invalid or incomplete" in app.session_state["generator_error"]
    assert app.session_state["results"] == RESULTS
    assert app.text_area(key="editor_0_0").value == RESULTS[0]["sql"]
    assert app.text_area(key="editor_0_1").value == RESULTS[1]["sql"]
    assert not any(button.label == "Run Generated Query" for button in app.button)


def test_generator_retries_transient_failure(gemini, parse_only):
    _, generate, delay = gemini
    generate.side_effect = [
        errors.APIError(429, {}), errors.APIError(503, {}), response(json.dumps(GENERATED)),
    ]
    assert ai.generate_sql("List customer IDs", {FRIENDLY: DATASET}).tables_used == [FRIENDLY]
    assert delay.call_args_list == [call(0.5), call(1.0)]
    parse_only.pgconn.prepare.assert_called_once()


def test_generator_handles_timeout_without_sql_validation(gemini, parse_only):
    import httpx2

    _, generate, delay = gemini
    generate.side_effect = httpx2.ReadTimeout("private response")
    with pytest.raises(ai.AdvisorError, match="Gemini timed out"):
        ai.generate_sql("List customer IDs", {FRIENDLY: DATASET})
    delay.assert_not_called()
    parse_only.pgconn.prepare.assert_not_called()
