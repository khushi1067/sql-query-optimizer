from __future__ import annotations
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from psycopg import Error as DatabaseError, OperationalError, errors, sql
from psycopg.rows import dict_row
from optimizer.config import SCHEMA, MAX_QUERIES, uploader_url
from optimizer.data import (
    DatasetError, create_dataset, dataset_sql_name, default_table_name, inspect_csv, normalize_headers, remove_dataset,
    suggested_relationships, validate_table_name,
)
from optimizer.query import PREVIEW_ROW_LIMIT, QueryError, compare, insights, run_query_preview
from optimizer.ai import AdvisorError, generate_sql, get_suggestion
from optimizer.report import render_report
from optimizer.db import ReaderConfigurationError, read_connection, workspace_indexes

st.set_page_config(page_title="SQL Query Optimizer", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: clamp(3.5rem, 7vw, 5rem);
    padding-bottom: 3rem;
}
.st-key-optimizer_intro {
    background: linear-gradient(115deg, #121c34, #263975);
    border: 1px solid #364985;
    border-radius: 1.25rem;
    padding: clamp(1.25rem, 4vw, 2.75rem);
    margin-bottom: .75rem;
}
.st-key-optimizer_intro h1 {
    color: #fff;
    font-size: clamp(2rem, 5vw, 3.3rem);
    letter-spacing: -.045em;
    padding: 0;
}
.st-key-optimizer_intro p {
    color: #dbe5ff;
    max-width: 52ch;
}
.st-key-dataset_panel, .st-key-generator_panel, .st-key-query_panel, .st-key-results_panel, .st-key-advisor_panel {
    border-radius: 1rem;
    padding: clamp(.9rem, 2.6vw, 1.6rem);
    margin-bottom: .6rem;
}
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    padding: 1rem;
    border-radius: .75rem;
}
[data-testid="stTextArea"] textarea {
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: .875rem;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)


def reset_results():
    for key in ["results", "ai_suggestion", "ai_result", "ai_baseline", "sample_consent"]:
        st.session_state.pop(key, None)


def install_dataset(dataset):
    sql_name = dataset_sql_name(dataset)
    workspace[sql_name] = dataset
    if st.session_state.get("queries"):
        return
    st.session_state["queries"] = [
        f"SELECT * FROM {sql_name} LIMIT 100",
        f"SELECT * FROM {sql_name} ORDER BY 1 LIMIT 100",
    ]
    st.session_state["generation"] = st.session_state.get("generation", 0) + 1
    reset_results()


def copy_sql_button(statement: str):
    # A JSON literal keeps the editor contents as data, including any closing HTML tags.
    encoded = json.dumps(statement).replace("<", "\\u003c")
    components.html(
        """<!doctype html>
<style>
  body { margin: 0; font: 14px system-ui; color-scheme: light dark; color: CanvasText; }
  button { font: inherit; padding: .4rem .8rem; cursor: pointer; }
  #status { margin-left: .4rem; }
</style>
<button id="copy" type="button">Copy SQL</button><span id="status" role="status"></span>
<script>
const sql = """ + encoded + """;
const status = document.getElementById("status");
document.getElementById("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(sql);
    status.textContent = "Copied.";
  } catch {
    const field = document.createElement("textarea");
    field.value = sql;
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.append(field);
    field.select();
    try {
      status.textContent = document.execCommand("copy") ? "Copied." : "Copy the SQL manually.";
    } catch {
      status.textContent = "Copy the SQL manually.";
    } finally {
      field.remove();
    }
  }
});
</script>""",
        height=45,
    )


def show_sql_generator(workspace, tables):
    with st.container(border=True, key="generator_panel"):
        st.header("AI SQL Generator")
        if not workspace:
            st.caption("Import a CSV to generate SQL.")
            return
        question = st.text_input(
            "Describe your query", key="generator_question",
            placeholder="Show me the top 10 customers by total spending in 2025", max_chars=2000,
        )
        st.caption("Gemini receives your question, table schemas, row counts, and suggested join keys.")
        if st.button("Generate SQL", disabled=not question.strip()):
            st.session_state.pop("generator_error", None)
            try:
                with st.spinner("Generating and validating SQL..."):
                    draft = generate_sql(question, workspace)
            except AdvisorError as exc:
                st.session_state["generator_error"] = str(exc)
            else:
                st.session_state["generator_sql"] = draft.generated_sql
                st.session_state["generator_explanation"] = draft.explanation
                st.session_state["generator_notes"] = (draft.assumptions, draft.warnings)
                st.session_state.pop("generator_results", None)

        if st.session_state.get("generator_explanation"):
            st.caption("Explanation of Gemini's draft")
            st.text(st.session_state["generator_explanation"])
            assumptions, warnings = st.session_state["generator_notes"]
            for assumption in assumptions:
                st.text("Assumption: " + assumption)
            for warning in warnings:
                st.text("Warning: " + warning)
            edited_sql = st.text_area("Generated SQL", key="generator_sql", height=175)
            copy, run = st.columns(2)
            if edited_sql.strip():
                with copy:
                    copy_sql_button(edited_sql)
            if run.button("Run Generated Query", disabled=not edited_sql.strip()):
                st.session_state.pop("generator_error", None)
                st.session_state.pop("generator_results", None)
                try:
                    with st.spinner("Running the generated query..."):
                        preview = run_query_preview(edited_sql, tables)
                except QueryError:
                    st.session_state["generator_error"] = "Use one read-only SELECT against workspace tables."
                except errors.UndefinedColumn:
                    st.session_state["generator_error"] = "The query references a column that does not exist."
                except errors.QueryCanceled:
                    st.session_state["generator_error"] = "The generated query timed out."
                except (DatabaseError, ReaderConfigurationError):
                    st.session_state["generator_error"] = "Could not run the generated query. Check the reader connection and SQL."
                else:
                    st.session_state["generator_results"] = (edited_sql, preview)

            result = st.session_state.get("generator_results")
            if result and result[0] == edited_sql:
                preview = result[1]
                headers = preview.columns
                if len(set(headers)) != len(headers):
                    headers = normalize_headers(headers)
                st.subheader("Generated query results")
                st.caption(f"Showing the first {PREVIEW_ROW_LIMIT} rows." if preview.truncated else f"{len(preview.rows):,} rows")
                st.dataframe(pd.DataFrame.from_records(preview.rows, columns=headers), hide_index=True)
        if st.session_state.get("generator_error"):
            st.error(st.session_state["generator_error"])


workspace = st.session_state.setdefault("workspace", {})
legacy_dataset = st.session_state.pop("dataset", None)
if legacy_dataset:
    workspace[dataset_sql_name(legacy_dataset)] = legacy_dataset

with st.container(key="optimizer_intro"):
    st.title("SQL Query Optimizer")
    st.write("Compare PostgreSQL execution plans and benchmark SQL queries on your own data.")

with st.container(border=True, key="dataset_panel"):
    st.header("Workspace tables")
    st.caption("Choose one or more CSVs. Per file: up to 10 MB, 50,000 rows, and 50 columns.")
    upload_configured = bool(uploader_url().strip())
    if not upload_configured:
        st.warning(
            "CSV preview is available. To import into PostgreSQL, provision the restricted "
            "database roles and set UPLOAD_DATABASE_URL in .env. See Local setup in README.md."
        )
    files = st.file_uploader(
        "Choose CSV files", type=["csv"], accept_multiple_files=True, key="workspace_uploads",
    )
    selected_ids = {file.file_id for file in files}
    imported_uploads = {
        file_id: name for file_id, name in st.session_state.get("imported_uploads", {}).items()
        if file_id in selected_ids and name in workspace
    }
    st.session_state["imported_uploads"] = imported_uploads
    for file in files:
        with st.container(border=True):
            st.text(file.name)
            if file.file_id in imported_uploads:
                st.success(f"Imported as {imported_uploads[file.file_id]}")
                continue
            table_name = st.text_input(
                "Table name", value=default_table_name(file.name), key=f"upload_name_{file.file_id}",
                help="Use up to 40 lowercase letters, digits, and underscores. Start with a letter.",
            )
            name_valid = True
            try:
                validate_table_name(table_name)
                if table_name in workspace:
                    raise DatasetError("That table name is already in this workspace.")
            except DatasetError as exc:
                name_valid = False
                st.error(str(exc))
            try:
                raw = file.getvalue()
                preview = inspect_csv(raw, file.name)
                st.success(f"Valid CSV: {len(preview.dataframe):,} rows and {len(preview.dataframe.columns)} columns")
                st.dataframe(preview.dataframe.head(10), use_container_width=True, hide_index=True)
                with st.expander("Detected column types"):
                    st.dataframe(pd.DataFrame([{"Column": k, "PostgreSQL type": v} for k,v in preview.pg_types.items()]),hide_index=True)
                if st.button(
                    "Import dataset into PostgreSQL", key=f"import_{file.file_id}",
                    type="primary", disabled=not upload_configured or not name_valid,
                ):
                    try:
                        dataset = create_dataset(preview, file.name, table_name)
                    except (errors.InsufficientPrivilege, errors.InvalidSchemaName,
                            errors.UndefinedObject, errors.ReadOnlySqlTransaction):
                        st.error(
                            "PostgreSQL rejected the import setup. Check that UPLOAD_DATABASE_URL "
                            "points to the optimizer_uploader role in the provisioned database "
                            "and that the optimizer_data schema and optimizer_reader role exist."
                        )
                    except errors.QueryCanceled:
                        st.error("The PostgreSQL import timed out. Try importing a smaller CSV.")
                    except OperationalError:
                        st.error(
                            "Could not connect to PostgreSQL. Check the credentials and host in "
                            "UPLOAD_DATABASE_URL and confirm the database is reachable."
                        )
                    except DatabaseError:
                        st.error("PostgreSQL rejected the import. Check the CSV data and database configuration.")
                    else:
                        install_dataset(dataset)
                        imported_uploads[file.file_id] = table_name
                        st.rerun()
            except DatasetError as exc:
                st.error(str(exc))

    for name, dataset in workspace.items():
        with st.expander(f"{name} · {dataset['rows']:,} rows · {len(dataset['columns'])} columns"):
            st.dataframe(pd.DataFrame(dataset["columns"], columns=["Column", "Type"]), hide_index=True)
            st.caption("First 10 rows")
            if "preview" in dataset:
                st.dataframe(dataset["preview"], hide_index=True)
            elif st.button("Load preview", key=f"preview_{name}"):
                try:
                    with read_connection() as conn:
                        cursor = conn.execute(sql.SQL("SELECT * FROM {} LIMIT 10").format(
                            sql.Identifier(SCHEMA, dataset["table"])
                        ))
                        dataset["preview"] = pd.DataFrame.from_records(
                            cursor.fetchall(), columns=[column.name for column in cursor.description],
                        )
                    st.rerun()
                except (DatabaseError, ReaderConfigurationError):
                    st.error("Could not load this table's preview.")
            confirmed = st.checkbox(f"Confirm deletion of {name}", key=f"confirm_delete_{name}")
            if st.button(f"Delete {name}", disabled=not confirmed):
                try:
                    remove_dataset(dataset["table"])
                except DatabaseError:
                    st.error("Could not delete this table.")
                else:
                    del workspace[name]
                    st.session_state.pop("generator_results", None)
                    for file_id, imported_name in list(imported_uploads.items()):
                        if imported_name == name:
                            del imported_uploads[file_id]
                    if not workspace:
                        st.session_state.pop("queries", None)
                    reset_results()
                    st.rerun()

    relationships = suggested_relationships(workspace)
    if len(workspace) > 1:
        with st.expander("Join keys and examples"):
            st.caption("Join tables with JOIN … ON. You can use any compatible columns as join keys.")
            if relationships:
                st.caption("Possible keys based on matching column names and types. Foreign keys and uniqueness are unverified.")
                st.dataframe(pd.DataFrame(relationships), hide_index=True)
                match = relationships[0]
                key = sql.Identifier(match["column"]).as_string()
                st.code(
                    f"SELECT * FROM {match['left_table']} AS a\n"
                    f"JOIN {match['right_table']} AS b ON a.{key} = b.{key}\nLIMIT 10",
                    language="sql",
                )
            else:
                st.caption("No columns share a name and type. Look for related columns with different names.")

tables = {name: dataset["table"] for name, dataset in workspace.items()}
sql_name = next(iter(workspace), None)

show_sql_generator(workspace, tables)

with st.container(border=True, key="query_panel"):
    st.header("SQL workspace")
    if workspace:
        st.caption("Compare two to five SELECT queries against your workspace tables.")
        queries = st.session_state.get("queries", [])
        generation = st.session_state.get("generation", 0)
        next_queries = []
        for i, query in enumerate(queries):
            column = i % 2
            if column == 0:
                editor_row = st.columns(2)
            with editor_row[column]:
                editor_key = f"editor_{generation}_{i}"
                st.session_state.setdefault(editor_key, query)
                value = st.text_area(f"Query {chr(65+i)}", height=175, key=editor_key)
                next_queries.append(value)
        st.session_state["queries"] = next_queries
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Add query", disabled=len(queries)>=MAX_QUERIES):
                st.session_state["queries"].append(f"SELECT * FROM {sql_name} LIMIT 100")
                st.rerun()
        with b2:
            if st.button("Remove last query", disabled=len(queries)<=2):
                st.session_state["queries"].pop()
                st.rerun()
        runs = st.radio("Benchmark repetitions", [5,1], horizontal=True,
                        format_func=lambda x: "5 runs (recommended)" if x==5 else "1 run (quick check)")
        if st.button("Analyze and compare", type="primary", use_container_width=True):
            try:
                with st.spinner("Running PostgreSQL benchmarks..."):
                    results = compare(st.session_state["queries"], tables, runs=runs)
                reset_results()
                st.session_state["results"] = results
                st.success("Comparison complete. Results are measured by PostgreSQL.")
            except (QueryError, ReaderConfigurationError) as exc:
                st.error(str(exc))
            except errors.QueryCanceled:
                st.error("A query exceeded the PostgreSQL statement timeout. Narrow the query and try again.")
            except (errors.UndefinedTable, errors.InsufficientPrivilege):
                st.error("The reader could not access the uploaded table. Check that both URLs point to the same database and optimizer_reader can select from the table.")
            except errors.UndefinedColumn:
                st.error("A query references a missing column. Check the workspace schemas for the imported column names.")
            except OperationalError:
                st.error("Could not connect to PostgreSQL. Check READONLY_DATABASE_URL and confirm the database is reachable.")
            except Exception:
                st.error("Query analysis failed. Check SQL syntax, query timeout, table name, and database connection.")
    else:
        st.caption("Import a CSV above to open the SQL editors.")

results = st.session_state.get("results")
if results and workspace:
    with st.container(border=True, key="results_panel"):
        st.header("Benchmark results")
        ranked = sorted(results, key=lambda x:x["median_ms"])
        best, slowest = ranked[0], ranked[-1]
        ratio = slowest["median_ms"]/best["median_ms"] if best["median_ms"] > 0 else None
        m1,m2,m3 = st.columns(3)
        m1.metric("Fastest measured", best["name"])
        m2.metric("Median execution", f"{best['median_ms']:.3f} ms")
        m3.metric("Speedup vs slowest", f"{ratio:.2f}×" if ratio is not None else "N/A")
        st.warning("Result equivalence: UNVERIFIED. Compare query results before adopting a faster rewrite.")
        chart_df = pd.DataFrame([{"Query":r["name"],"Execution (ms)":r["median_ms"]} for r in results])
        fig = px.bar(chart_df, x="Query", y="Execution (ms)", text_auto=".3f",
                     color="Query", color_discrete_sequence=["#6366f1","#10b981","#06b6d4","#f59e0b","#8b5cf6"])
        fig.update_layout(
            showlegend=False, height=320, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title=None, yaxis_title="Median execution (ms)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame([{"Query":r["name"],"Median ms":r["median_ms"],
              "Min ms":r["min_ms"],"Max ms":r["max_ms"],
              "Planning ms":r["metrics"]["planning_ms"],
              "Shared hits":r["metrics"]["buffer_hits"],"Shared reads":r["metrics"]["buffer_reads"],
              "Join operations": ", ".join(node["node"] for node in r["metrics"].get("join_operations", []))}
             for r in results]),hide_index=True,use_container_width=True)
        for r in results:
            with st.expander(f"{r['name']}: execution-plan insights"):
                for observation in insights(r["metrics"]):
                    st.write("• "+observation)
                st.dataframe(pd.DataFrame(r["metrics"]["nodes"]),hide_index=True,use_container_width=True)
                st.caption("Plan nodes shown in traversal order; depth indicates parent/child nesting.")

    with st.container(border=True, key="advisor_panel"):
        st.header("AI optimization advisor")
        st.info("Your queries, schema and summarized performance data will be sent to Google Gemini. No raw CSV rows are sent by default. Queries themselves can contain sensitive literals.")
        consent = st.checkbox("I agree to share the above SQL and performance metadata with Google Gemini", key="ai_consent")
        include_rows = st.checkbox("Also share up to 10 sample rows with Google Gemini", key="include_rows")
        sample_table = st.selectbox("Table to sample", list(workspace)) if include_rows else None
        if st.button("Ask AI to optimize", disabled=not consent):
            stage = "request"
            try:
                samples = None
                if sample_table is not None:
                    stage = "sample_loading"
                    with read_connection() as conn:
                        with conn.cursor(row_factory=dict_row) as cursor:
                            cursor.execute(sql.SQL("SELECT * FROM {} LIMIT 10").format(sql.Identifier(SCHEMA, tables[sample_table])))
                            samples = {sample_table: cursor.fetchall()}
                stage = "request"
                with st.spinner("AI is reviewing PostgreSQL performance context..."):
                    try:
                        indexes = workspace_indexes(tables)
                    except (DatabaseError, ReaderConfigurationError):
                        indexes = None
                    suggestion = get_suggestion(workspace, None, best, results, samples, indexes=indexes)
                stage = "ui_state"
                st.session_state["ai_suggestion"] = suggestion
                st.session_state.pop("ai_result", None)
                st.session_state.pop("ai_baseline", None)
            except AdvisorError as exc:
                st.error(str(exc))
            except Exception as exc:
                message = {
                    "sample_loading": "Could not load sample rows. Turn off sample sharing to try again.",
                    "request": "Could not prepare the AI request. Rerun the query comparison and try again.",
                    "ui_state": "Could not save the AI suggestion to this dashboard session.",
                }[stage]
                st.error(str(AdvisorError(stage, stage + "_failed", message, cause=exc)))
        suggestion = st.session_state.get("ai_suggestion")
        if suggestion and suggestion.no_safe_optimization:
            st.info("No safe optimization recommended.")
            st.write(suggestion.explanation)
            st.caption(suggestion.expected_mechanism)
            st.warning(suggestion.cautions)
        elif suggestion:
            st.markdown("**AI-suggested query**")
            st.code(suggestion.optimized_sql, language="sql")
            st.markdown("**Why this might improve performance**")
            st.write(suggestion.explanation)
            st.caption("Optimization hypothesis: "+suggestion.expected_mechanism)
            st.warning("Cautions: "+suggestion.cautions+" AI-generated SQL may change semantics. Verify results independently.")
            if st.button("Test AI Query"):
                try:
                    with st.spinner("Benchmarking candidate on PostgreSQL..."):
                        trial = compare(
                            [best["sql"], suggestion.optimized_sql], tables, runs=5,
                        )
                    st.session_state["ai_result"] = trial[1]
                    st.session_state["ai_baseline"] = trial[0]
                except Exception:
                    st.error("The suggested SQL could not be benchmarked safely.")
            ai_result = st.session_state.get("ai_result")
            if ai_result:
                baseline = st.session_state["ai_baseline"]
                ca,cb = st.columns(2)
                ca.metric("Retested original",f"{baseline['median_ms']:.3f} ms")
                cb.metric("AI candidate",f"{ai_result['median_ms']:.3f} ms")
                if ai_result["median_ms"] < baseline["median_ms"]:
                    st.success("The AI candidate was faster in this benchmark. Equivalence is still unverified.")
                else:
                    st.info("The AI candidate was not faster in this benchmark.")

    report_html = render_report(
        next(iter(workspace.values())), results, st.session_state.get("ai_suggestion"),
        st.session_state.get("ai_result"), workspace=workspace, ai_baseline=st.session_state.get("ai_baseline"),
    )
    st.download_button("Download HTML report",report_html,file_name="sql_optimization_report.html",mime="text/html",use_container_width=True)

st.divider()
st.caption("Local single-user learning tool · read-only SQL execution · not designed for public multi-tenant hosting without stronger identity and isolation controls.")
