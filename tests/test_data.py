import pytest
from optimizer.data import inspect_csv, normalize_headers, DatasetError, infer_types
import pandas as pd


def test_valid_csv():
    preview = inspect_csv(b"id,amount,day\n1,2.50,2025-01-01\n2,3.00,2025-01-02\n", "sales.csv")
    assert preview.pg_types == {"id": "BIGINT", "amount": "NUMERIC", "day": "DATE"}
    assert len(preview.dataframe) == 2


def test_headers_normalized():
    assert normalize_headers(["Order ID", "Order ID", "123 points"]) == ["order_id", "order_id_2", "col_123_points"]


@pytest.mark.parametrize("filename,data", [
    ("orders.xlsx", b"col\n1\n"),
    ("orders.csv", b""),
    ("orders.csv", b"id,id\n1,2\n"),
    ("orders.csv", b"\xff\xfe"),
    ("orders.csv", b"col\n"),
])
def test_invalid_uploads(filename,data):
    with pytest.raises(DatasetError):
        inspect_csv(data,filename)


def test_leading_zero_identifier_stays_text():
    frame = pd.DataFrame({"zip": ["00123", "00456"]})
    assert infer_types(frame)["zip"] == "TEXT"


@pytest.mark.parametrize("name", ["customers", "sales2026", "employee_data", "a" * 40])
def test_valid_table_names(name):
    from optimizer.data import validate_table_name

    assert validate_table_name(name) == name


@pytest.mark.parametrize("name", [
    "", "Customers", "_sales", "1sales", "sales orders", "sales;drop", "public.sales",
    "select", "current_user", "pg_class", "optimizer_data", "a" * 41, "ds_" + "a" * 32,
])
def test_invalid_table_names(name):
    from optimizer.data import validate_table_name

    with pytest.raises(DatasetError):
        validate_table_name(name)


@pytest.mark.parametrize(("filename", "name"), [
    ("Customer Orders.CSV", "customer_orders"),
    ("2026 Sales.csv", "dataset_2026_sales"),
    ("Select.csv", "data_select"),
    ("!.csv", "dataset"),
])
def test_table_name_defaults(filename, name):
    from optimizer.data import default_table_name

    assert default_table_name(filename) == name


def test_same_friendly_name_has_distinct_physical_tables(monkeypatch):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    from optimizer import db
    from optimizer.data import create_dataset

    conn = MagicMock()

    @contextmanager
    def connect():
        yield conn

    monkeypatch.setattr(db, "upload_connection", connect)
    preview = inspect_csv(b"id\n1\n", "input.csv")
    first = create_dataset(preview, "input.csv", "customers")
    second = create_dataset(preview, "input.csv", "customers")
    assert first["friendly_name"] == second["friendly_name"] == "customers"
    assert first["table"] != second["table"]
    assert first["table"] in conn.execute.call_args_list[0].args[0].as_string()
    assert second["table"] in conn.execute.call_args_list[2].args[0].as_string()


def test_upload_uses_custom_name_without_removing_previous_dataset(monkeypatch):
    from pathlib import Path
    from unittest.mock import Mock

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from optimizer import config, data

    uploaded = Mock()
    uploaded.file_id = "customer-orders"
    uploaded.name = "Customer Orders.csv"
    uploaded.getvalue.return_value = b"id\n1\n"
    monkeypatch.setattr(st, "file_uploader", Mock(return_value=[uploaded]))
    monkeypatch.setattr(config, "uploader_url", lambda: "postgresql://fixture")
    dataset = {
        "table": "ds_" + "a" * 32, "qualified": "optimizer_data.ds_" + "a" * 32,
        "display_name": uploaded.name, "rows": 1, "columns": [("id", "BIGINT")],
    }
    create = Mock(return_value={**dataset, "friendly_name": "customers"})
    remove = Mock()
    monkeypatch.setattr(data, "create_dataset", create)
    monkeypatch.setattr(data, "remove_dataset", remove)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["dataset"] = {**dataset, "table": "ds_" + "b" * 32, "friendly_name": "previous"}
    app.run()

    def import_button():
        return next(button for button in app.button if button.label.startswith("Import dataset"))

    assert app.text_input[0].value == "customer_orders"
    app.text_input[0].set_value("select").run()
    assert import_button().disabled
    app.text_input[0].set_value("customers").run()
    import_button().click().run()
    assert not app.exception
    assert create.call_args.args[1:] == (uploaded.name, "customers")
    assert set(app.session_state["workspace"].keys()) == {"previous", "customers"}
    assert app.text_area[0].value == "SELECT * FROM customers LIMIT 100"
    remove.assert_not_called()


def test_dashboard_four_file_batch_preserves_workspace_and_supports_confirmed_delete(monkeypatch):
    from pathlib import Path
    from unittest.mock import Mock

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from optimizer import config, data

    fixtures = {
        "customers": b"customer_id,region\n1,East\n",
        "orders": b"order_id,customer_id\n10,1\n",
        "products": b"product_id,sku\n5,A\n",
        "order_items": b"order_id,product_id,quantity\n10,5,2\n",
    }
    files = []
    for name, content in fixtures.items():
        uploaded = Mock()
        uploaded.file_id = name
        uploaded.name = name + ".csv"
        uploaded.getvalue.return_value = content
        files.append(uploaded)
    uploader = Mock(return_value=files)
    monkeypatch.setattr(st, "file_uploader", uploader)
    monkeypatch.setattr(config, "uploader_url", lambda: "postgresql://fixture")

    def persist(preview, filename, name):
        physical = "ds_" + format(list(fixtures).index(name), "032x")
        return {
            "table": physical, "friendly_name": name, "display_name": filename,
            "rows": len(preview.dataframe), "columns": list(preview.pg_types.items()),
            "preview": preview.dataframe.copy(),
        }

    create = Mock(side_effect=persist)
    delete = Mock()
    monkeypatch.setattr(data, "create_dataset", create)
    monkeypatch.setattr(data, "remove_dataset", delete)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))

    def button(label):
        return next(item for item in app.button if item.label == label)

    app.run()
    assert uploader.call_args.kwargs["accept_multiple_files"] is True
    assert [field.value for field in app.text_input] == list(fixtures)

    app.button(key="import_customers").click().run()
    original = app.session_state["workspace"]["customers"]
    app.text_input(key="upload_name_orders").set_value("customers").run()
    assert app.button(key="import_orders").disabled
    app.text_input(key="upload_name_orders").set_value("orders").run()

    for name in ("orders", "products", "order_items"):
        app.button(key=f"import_{name}").click().run()
        assert not app.exception
    assert set(app.session_state["workspace"]) == set(fixtures)
    assert app.session_state["workspace"]["customers"] is original
    assert create.call_count == 4
    assert len([status for status in app.success if status.value.startswith("Imported as")]) == 4

    uploader.return_value = []
    app.run()
    assert set(app.session_state["workspace"]) == set(fixtures)
    assert button("Delete orders").disabled
    assert len(data.suggested_relationships(app.session_state["workspace"])) == 3
    next(item for item in app.checkbox if item.label == "Confirm deletion of orders").check().run()
    button("Delete orders").click().run()
    delete.assert_called_once_with("ds_" + format(1, "032x"))
    assert set(app.session_state["workspace"]) == {"customers", "products", "order_items"}
    assert app.text_area[0].value == "SELECT * FROM customers LIMIT 100"


def test_workspace_index_metadata_is_scoped_and_omits_physical_identifiers(monkeypatch):
    from contextlib import contextmanager
    from unittest.mock import Mock

    from optimizer import db

    physical = "ds_" + "a" * 32
    other = "ds_" + "b" * 32
    conn = Mock()
    conn.execute.return_value.fetchall.return_value = [
        (physical, "btree", True, False, ["customer_id", None]),
    ]

    @contextmanager
    def connect():
        yield conn

    monkeypatch.setattr(db, "read_connection", connect)
    assert db.workspace_indexes({"sales": physical, "customers": other}) == {
        "sales": [{"method": "btree", "unique": True, "partial": False,
                   "key_columns": ["customer_id", "(expression)"]}],
        "customers": [],
    }
    assert conn.execute.call_args.args[1] == ("optimizer_data", [physical, other])
