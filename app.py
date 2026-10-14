import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import os
import requests
import altair as alt
import pandas as pd

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
HF_MODELS_API_URL = "https://huggingface.co/api/models"
OPENML_FLOWS_API_URL = "https://www.openml.org/api/v1/json/flow/list"
OPENML_FLOW_PAGE_URL = "https://www.openml.org/f/"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cisa_kev():
    response = requests.get(CISA_KEV_URL, timeout=15)
    response.raise_for_status()
    catalog = response.json()
    vulnerabilities = catalog.get("vulnerabilities") if isinstance(catalog, dict) else None
    if not isinstance(vulnerabilities, list):
        raise ValueError("CISA returned an unexpected catalog format.")
    return catalog.get("catalogVersion", "unknown"), vulnerabilities


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hf_models(limit=2000):
    params = {"limit": int(limit), "full": "true", "config": "true"}
    response = requests.get(HF_MODELS_API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Hugging Face returned an unexpected model catalog format.")
    return data


def get_openml_api_key():
    """Read the OpenML key from st.secrets, falling back to the environment."""
    try:
        key = st.secrets.get("OPENML_API_KEY")
    except Exception:  # no secrets file configured
        key = None
    key = key or os.getenv("OPENML_API_KEY")
    return str(key).strip() if key else None


@st.cache_resource(show_spinner=False)
def get_kaggle_api():
    """Authenticate once per process; credentials must be set before importing kaggle."""
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
        os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
    except (KeyError, FileNotFoundError) as exc:
        raise ValueError(f"Kaggle credentials are not configured in Streamlit secrets: {exc}") from None
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kaggle_models(limit=100):
    api = get_kaggle_api()
    try:
        models = api.model_list(page_size=int(limit)) or []
    except Exception as exc:
        raise ValueError(f"Kaggle request failed ({type(exc).__name__}).") from None
    records = []
    for model in models:
        if model is None:
            continue
        records.append(
            {
                name: getattr(model, name, None)
                for name in (
                    "ref", "author", "title", "subtitle", "description", "tags",
                    "vote_count", "publish_time", "update_time", "is_private", "url",
                )
            }
        )
    return records


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openml_models(limit=2000):
    api_key = get_openml_api_key()
    if not api_key:
        raise ValueError(
            "OpenML API key is not configured. Set OPENML_API_KEY in Streamlit secrets "
            "or as an environment variable."
        )
    url = f"{OPENML_FLOWS_API_URL}/limit/{int(limit)}"
    try:
        response = requests.get(url, params={"api_key": api_key}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        # Do not chain/echo the exception: its text can contain the request URL and key.
        status = getattr(exc.response, "status_code", None)
        detail = f"HTTP {status}" if status else type(exc).__name__
        raise ValueError(f"OpenML request failed ({detail}).") from None
    except ValueError:
        raise ValueError("OpenML returned a response that is not valid JSON.") from None
    flows = data.get("flows", {}).get("flow") if isinstance(data, dict) else None
    if isinstance(flows, dict):
        flows = [flows]
    if not isinstance(flows, list):
        raise ValueError("OpenML returned an unexpected model catalog format.")
    return flows


def selected_chart_values(widget_key, selection_name, field_name):
    widget_state = st.session_state.get(widget_key, {})
    selection_state = widget_state.get("selection", {})
    selected = selection_state.get(selection_name, [])
    if isinstance(selected, dict):
        selected = [selected]
    if not isinstance(selected, list):
        return set()
    return {
        item[field_name]
        for item in selected
        if isinstance(item, dict) and field_name in item
    }


def render_cisa_dashboard():
    st.title("CISA KEV Intelligence")
    st.caption("Explore vulnerabilities CISA has confirmed as known exploited.")

    if st.sidebar.button("Refresh CISA data", width="stretch"):
        fetch_cisa_kev.clear()

    try:
        with st.spinner("Loading the CISA KEV catalog..."):
            catalog_version, vulnerabilities = fetch_cisa_kev()
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Could not load CISA KEV data: {exc}")
        return

    records = pd.DataFrame(vulnerabilities)
    if records.empty:
        st.warning("The CISA catalog contains no vulnerability records.")
        return

    for column in (
        "cveID",
        "vulnerabilityName",
        "vendorProject",
        "product",
        "dateAdded",
        "knownRansomwareCampaignUse",
        "cwes",
        "shortDescription",
        "requiredAction",
    ):
        if column not in records:
            records[column] = None

    records["dateAdded"] = pd.to_datetime(records["dateAdded"], errors="coerce")
    records["cwes"] = records["cwes"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    records["ransomwareKnown"] = (
        records["knownRansomwareCampaignUse"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .eq("known")
    )
    latest_date = records["dateAdded"].max()
    if pd.isna(latest_date):
        st.error("The CISA catalog does not contain valid date-added values.")
        return

    st.caption(
        f"Source: [CISA KEV catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) "
        f"· Catalog version {catalog_version} · {len(records):,} records · Cached for up to 1 hour."
    )

    cwe_options = sorted({cwe for values in records["cwes"] for cwe in values})
    filter_columns = st.columns([1, 1.5, 1.5, 1.5, 2])
    period = filter_columns[0].selectbox(
        "Catalog lookback",
        ["30 days", "90 days", "12 months", "All records"],
        index=1,
    )
    vendors = sorted(records["vendorProject"].dropna().astype(str).unique())
    selected_vendors = filter_columns[1].multiselect(
        "Vendor",
        vendors,
        placeholder="All vendors",
    )
    selected_cwes = filter_columns[2].multiselect(
        "CWE",
        cwe_options,
        placeholder="All CWEs",
    )
    ransomware_filter = filter_columns[3].selectbox(
        "Ransomware use",
        ["All", "Known", "Not marked known"],
    )
    search_query = filter_columns[4].text_input("Search records")

    filtered = records.copy()
    lookback_days = {"30 days": 30, "90 days": 90, "12 months": 365}
    if period in lookback_days:
        start_date = latest_date - pd.Timedelta(days=lookback_days[period] - 1)
        filtered = filtered[filtered["dateAdded"] >= start_date]
    if selected_vendors:
        filtered = filtered[filtered["vendorProject"].isin(selected_vendors)]
    if selected_cwes:
        selected_cwe_set = set(selected_cwes)
        filtered = filtered[
            filtered["cwes"].apply(lambda values: bool(selected_cwe_set.intersection(values)))
        ]
    if ransomware_filter == "Known":
        filtered = filtered[filtered["ransomwareKnown"]]
    elif ransomware_filter == "Not marked known":
        filtered = filtered[~filtered["ransomwareKnown"]]
    if search_query.strip():
        searchable = filtered[
            ["cveID", "vulnerabilityName", "vendorProject", "product", "shortDescription"]
        ].fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered[
            searchable.str.contains(search_query.strip(), case=False, regex=False)
        ]

    chart_records = filtered.copy()
    chart_vendors = selected_chart_values(
        "kev_vendor_chart",
        "vendor_filter",
        "vendorProject",
    )
    chart_cwes = selected_chart_values("kev_cwe_chart", "cwe_filter", "cwes")
    if chart_vendors:
        filtered = filtered[filtered["vendorProject"].isin(chart_vendors)]
    if chart_cwes:
        filtered = filtered[
            filtered["cwes"].apply(lambda values: bool(chart_cwes.intersection(values)))
        ]

    known_ransomware_count = int(filtered["ransomwareKnown"].sum())
    metric_columns = st.columns(4)
    metric_columns[0].metric("Catalog entries", f"{len(records):,}")
    metric_columns[1].metric("Matching filters", f"{len(filtered):,}")
    metric_columns[2].metric("Known ransomware use", f"{known_ransomware_count:,}")
    metric_columns[3].metric(
        "Vendors represented",
        f"{filtered['vendorProject'].nunique():,}",
    )

    if filtered.empty:
        st.info("No records match these filters.")
        return

    trend_data = (
        filtered.dropna(subset=["dateAdded"])
        .assign(month=lambda frame: frame["dateAdded"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "entries"})
    )
    trend_chart = (
        alt.Chart(trend_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Date added"),
            y=alt.Y("entries:Q", title="Catalog additions"),
            tooltip=[alt.Tooltip("month:T", title="Month"), alt.Tooltip("entries:Q", title="Entries")],
        )
        .interactive()
        .properties(height=250)
    )
    st.subheader("Catalog additions over time")
    st.altair_chart(trend_chart, width="stretch")

    vendor_counts = (
        chart_records.groupby("vendorProject", dropna=True)
        .size()
        .nlargest(10)
        .rename("entries")
        .reset_index()
    )
    cwe_counts = (
        chart_records.explode("cwes")
        .dropna(subset=["cwes"])
        .groupby("cwes")
        .size()
        .nlargest(10)
        .rename("entries")
        .reset_index()
    )
    vendor_column, cwe_column = st.columns(2)
    with vendor_column:
        st.subheader("Top vendors")
        if not vendor_counts.empty:
            vendor_selection = alt.selection_point(
                name="vendor_filter",
                fields=["vendorProject"],
                toggle=True,
            )
            vendor_chart = (
                alt.Chart(vendor_counts)
                .mark_bar()
                .encode(
                    x=alt.X("entries:Q", title="Entries"),
                    y=alt.Y("vendorProject:N", sort="-x", title=None),
                    tooltip=["vendorProject:N", "entries:Q"],
                    color=alt.condition(
                        vendor_selection,
                        alt.value("#087f73"),
                        alt.value("#9abcb7"),
                    ),
                )
                .add_params(vendor_selection)
                .properties(height=300)
            )
            st.altair_chart(
                vendor_chart,
                width="stretch",
                key="kev_vendor_chart",
                on_select="rerun",
                selection_mode="vendor_filter",
            )
        else:
            st.caption("No vendor data in these records.")
    with cwe_column:
        st.subheader("Top CWEs")
        if not cwe_counts.empty:
            cwe_selection = alt.selection_point(
                name="cwe_filter",
                fields=["cwes"],
                toggle=True,
            )
            cwe_chart = (
                alt.Chart(cwe_counts)
                .mark_bar()
                .encode(
                    x=alt.X("entries:Q", title="Entries"),
                    y=alt.Y("cwes:N", sort="-x", title=None),
                    tooltip=["cwes:N", "entries:Q"],
                    color=alt.condition(
                        cwe_selection,
                        alt.value("#bd5b26"),
                        alt.value("#d9a47f"),
                    ),
                )
                .add_params(cwe_selection)
                .properties(height=300)
            )
            st.altair_chart(
                cwe_chart,
                width="stretch",
                key="kev_cwe_chart",
                on_select="rerun",
                selection_mode="cwe_filter",
            )
        else:
            st.caption("No CWE data in these records.")

    st.subheader("Matching vulnerabilities")
    table = filtered.sort_values("dateAdded", ascending=False).reset_index(drop=True)
    table_view = table[
        [
            "cveID",
            "vulnerabilityName",
            "vendorProject",
            "product",
            "dateAdded",
            "knownRansomwareCampaignUse",
        ]
    ].rename(
        columns={
            "cveID": "CVE",
            "vulnerabilityName": "Vulnerability",
            "vendorProject": "Vendor",
            "product": "Product",
            "dateAdded": "Date added",
            "knownRansomwareCampaignUse": "Ransomware use",
        }
    )
    table_selection = st.dataframe(
        table_view,
        column_config={
            "Date added": st.column_config.DateColumn("Date added", format="YYYY-MM-DD"),
        },
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        height=420,
        width="stretch",
        key="cisa_findings_table",
    )
    st.download_button(
        "Download filtered records as CSV",
        data=table_view.to_csv(index=False),
        file_name="cisa_kev_filtered.csv",
        mime="text/csv",
    )

    selected_rows = table_selection.selection.rows
    if selected_rows and selected_rows[0] < len(table):
        selected = table.iloc[selected_rows[0]]
        st.subheader(f"{selected['cveID']} · {selected['vulnerabilityName']}")
        st.markdown(f"[Open CVE record](https://nvd.nist.gov/vuln/detail/{selected['cveID']})")
        st.write(selected["shortDescription"] or "No description provided.")
        st.markdown("**CISA required action**")
        st.write(selected["requiredAction"] or "No required action provided.")

UNKNOWN_LICENSES = ["unknown", "other", ""]

# Boolean feature column -> (slider label, default weight)
RISK_WEIGHT_CONTROLS = {
    "is_unknown_license": ("Unknown/other license", 25),
    "is_missing_card": ("Missing model-card metadata", 25),
    "is_stale": ("Stale (>365 days)", 20),
    "is_gated": ("Gated access", 10),
    "is_disabled": ("Disabled", 20),
    "is_high_adoption_low_transparency": ("High adoption + low transparency", 20),
}
RISK_FEATURES = list(RISK_WEIGHT_CONTROLS)


def _safe_count(value):
    if isinstance(value, bool):
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return int(number) if pd.notna(number) and number >= 0 else 0


def hf_model_url(row):
    return f"https://huggingface.co/{row['model_id']}" if row["model_id"] else ""


def openml_model_url(row):
    return f"{OPENML_FLOW_PAGE_URL}{row['openml_id']}" if row.get("openml_id") else ""


def kaggle_model_url(row):
    if row.get("kaggle_url"):
        return row["kaggle_url"]
    return f"https://www.kaggle.com/models/{row['model_id']}" if row["model_id"] else ""


def _finalize_model_frame(rows, columns, url_builder):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
    df["lastModified"] = pd.to_datetime(df["lastModified"], errors="coerce", utc=True)
    df["days_since_update"] = (
        (pd.Timestamp.now(tz="UTC") - df["lastModified"]).dt.days.fillna(99999).clip(lower=0)
    ).astype(int)

    # Baseline feature engineering only; scoring is applied by apply_risk_scores().
    df["is_unknown_license"] = df["license"].str.casefold().isin(UNKNOWN_LICENSES)
    df["is_missing_card"] = ~df["has_model_card_like"]
    df["is_stale"] = df["days_since_update"] > 365
    df["is_gated"] = df["gated"]
    df["is_disabled"] = df["disabled"]
    df["is_high_adoption_low_transparency"] = (
        df["downloads"] >= df["downloads"].quantile(0.90)
    ) & (df["is_unknown_license"] | df["is_missing_card"])
    df["model_url"] = [url_builder(row) for row in df.to_dict("records")]
    return df[columns]


def _model_columns():
    return [
        "model_id", "author", "pipeline_tag", "library_name", "downloads", "likes",
        "license", "createdAt", "lastModified", "tags", "has_model_card_like",
        "private", "gated", "disabled", "days_since_update", *RISK_FEATURES,
        "model_url",
    ]


def normalize_hf_models(raw_models):
    columns = _model_columns()
    rows = []
    for item in raw_models or []:
        if not isinstance(item, dict):
            continue
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        tags = [str(tag) for tag in tags if tag is not None]
        model_id = str(item.get("id") or item.get("modelId") or "")
        author = item.get("author") or (model_id.split("/", 1)[0] if "/" in model_id else None)
        license_value = "unknown"
        for tag in tags:
            if tag.startswith("license:"):
                license_value = tag.split("license:", 1)[1].strip() or "unknown"
                break
        rows.append(
            {
                "model_id": model_id,
                "author": str(author or "unknown"),
                "pipeline_tag": str(item.get("pipeline_tag") or "unknown"),
                "library_name": str(item.get("library_name") or "unknown"),
                "downloads": _safe_count(item.get("downloads")),
                "likes": _safe_count(item.get("likes")),
                "license": license_value,
                "createdAt": item.get("createdAt"),
                "lastModified": item.get("lastModified"),
                "tags": tags,
                "has_model_card_like": any(tag.startswith("model-index") for tag in tags)
                or bool(item.get("cardData")),
                "private": bool(item.get("private", False)),
                "gated": item.get("gated") not in (False, None, ""),
                "disabled": bool(item.get("disabled", False)),
            }
        )

    return _finalize_model_frame(rows, columns, hf_model_url)


def normalize_openml_models(raw_models):
    columns = _model_columns()
    rows = []
    for item in raw_models or []:
        if not isinstance(item, dict):
            continue
        flow_id = item.get("id")
        name = str(item.get("full_name") or item.get("name") or "").strip()
        if not name and flow_id is None:
            continue
        tags = item.get("tag") if "tag" in item else item.get("tags")
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(tag) for tag in tags if tag is not None] if isinstance(tags, list) else []
        library = str(item.get("library_name") or "").strip()
        if not library and "." in name:
            library = name.split(".", 1)[0]
        license_value = str(item.get("licence") or item.get("license") or "unknown").strip() or "unknown"
        rows.append(
            {
                "model_id": name or str(flow_id),
                "openml_id": str(flow_id) if flow_id is not None else "",
                "author": str(item.get("uploader") or "unknown"),
                "pipeline_tag": str(item.get("task_type") or "unknown"),
                "library_name": library or "unknown",
                "downloads": _safe_count(item.get("downloads")),
                "likes": 0,
                "license": license_value,
                "createdAt": item.get("upload_date"),
                "lastModified": item.get("upload_date"),
                "tags": tags,
                "has_model_card_like": bool(item.get("description")),
                "private": False,
                "gated": False,
                "disabled": False,
            }
        )
    return _finalize_model_frame(rows, columns, openml_model_url)


def normalize_kaggle_models(raw_models):
    columns = _model_columns()
    rows = []
    for item in raw_models or []:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        if not ref:
            continue
        tags = item.get("tags")
        tags = [str(getattr(t, "name", t)) for t in tags if t is not None] if isinstance(tags, list) else []
        rows.append(
            {
                "model_id": ref,
                "author": str(item.get("author") or (ref.split("/", 1)[0] if "/" in ref else "unknown")),
                "pipeline_tag": "unknown",
                "library_name": "unknown",
                "downloads": 0,
                "likes": _safe_count(item.get("vote_count")),
                "license": "unknown",
                "createdAt": item.get("publish_time"),
                "lastModified": item.get("update_time") or item.get("publish_time"),
                "tags": tags,
                "has_model_card_like": bool(item.get("description") or item.get("subtitle")),
                "private": bool(item.get("is_private", False)),
                "gated": False,
                "disabled": False,
                "kaggle_url": str(item.get("url") or ""),
            }
        )
    return _finalize_model_frame(rows, columns, kaggle_model_url)


def apply_risk_scores(models, weights):
    """Return a copy of models with risk_score (clipped 0-100) and risk_band."""
    scored = models.copy()
    raw_risk = pd.Series(0, index=scored.index, dtype="int64")
    for feature, weight in weights.items():
        if feature in scored.columns:
            raw_risk = raw_risk + scored[feature].fillna(False).astype(bool).astype(int) * int(weight)
    scored["risk_score"] = raw_risk.clip(0, 100).astype(int)
    scored["risk_band"] = pd.cut(
        scored["risk_score"], bins=[-1, 29, 59, 100], labels=["Low", "Medium", "High"]
    ).astype(str)
    return scored


def render_risk_weight_controls():
    with st.sidebar.expander("Risk weights", expanded=False):
        st.caption("Tune the heuristic; scores and bands recompute live (clipped to 0–100).")
        weights = {
            feature: st.slider(label, 0, 50, default, key=f"risk_weight_{feature}")
            for feature, (label, default) in RISK_WEIGHT_CONTROLS.items()
        }
        if st.button("Reset risk weights", width="stretch"):
            for feature in RISK_WEIGHT_CONTROLS:
                st.session_state.pop(f"risk_weight_{feature}", None)
            st.rerun()
    return weights


def load_hf_models():
    return normalize_hf_models(fetch_hf_models())


def load_openml_models():
    return normalize_openml_models(fetch_openml_models())


def load_kaggle_models():
    return normalize_kaggle_models(fetch_kaggle_models())


# Provider registry: label -> loader returning a normalized frame (None = coming soon).
# New providers should supply a fetch + normalize pair producing the same columns.
MODEL_HUB_PROVIDERS = {
    "Hugging Face": load_hf_models,
    "OpenML": load_openml_models,
    "Kaggle": load_kaggle_models,
}

# Provider label -> (display name, docs URL, cached fetch function to clear on refresh).
MODEL_HUB_SOURCES = {
    "Hugging Face": ("Hugging Face", "https://huggingface.co/docs/hub/api", fetch_hf_models),
    "OpenML": ("OpenML", "https://www.openml.org/apis", fetch_openml_models),
    "Kaggle": ("Kaggle", "https://www.kaggle.com/docs/api", fetch_kaggle_models),
}


def render_model_hub_dashboard():
    st.title("Model Hub Intelligence")
    provider = st.radio("Provider", list(MODEL_HUB_PROVIDERS), horizontal=True)
    loader = MODEL_HUB_PROVIDERS[provider]
    if loader is None:
        st.info(
            f"{provider.split(' (')[0]} support is planned. "
            "Select Hugging Face to explore model metadata today."
        )
        return

    source_label, source_url, refresh_fn = MODEL_HUB_SOURCES[provider]
    st.caption(
        f"Explore adoption, transparency, risk, and operational signals from public {source_label} model metadata."
    )

    if st.sidebar.button("Refresh model hub data", width="stretch"):
        refresh_fn.clear()
    weights = render_risk_weight_controls()

    try:
        with st.spinner(f"Loading the {source_label} model catalog..."):
            models = loader()
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Could not load {source_label} model data: {exc}")
        return

    if models.empty:
        st.warning(f"The {source_label} API returned no model records.")
        return

    models = apply_risk_scores(models, weights)

    st.caption(
        f"Source: [{source_label} API]({source_url}) "
        f"· {len(models):,} models · Cached for up to 1 hour."
    )

    filter_columns = st.columns([1.3, 1.3, 1.3, 1, 1.2, 2])
    selected_tasks = filter_columns[0].multiselect(
        "Task", sorted(models["pipeline_tag"].unique()), placeholder="All tasks"
    )
    selected_libraries = filter_columns[1].multiselect(
        "Library", sorted(models["library_name"].unique()), placeholder="All libraries"
    )
    selected_licenses = filter_columns[2].multiselect(
        "License", sorted(models["license"].unique()), placeholder="All licenses"
    )
    risk_filter = filter_columns[3].selectbox("Risk band", ["All", "Low", "Medium", "High"])
    recency_filter = filter_columns[4].selectbox(
        "Update recency",
        ["Any", "≤ 30 days", "≤ 90 days", "≤ 365 days", "> 365 days"],
    )
    search_query = filter_columns[5].text_input("Search models")

    filtered = models.copy()
    if selected_tasks:
        filtered = filtered[filtered["pipeline_tag"].isin(selected_tasks)]
    if selected_libraries:
        filtered = filtered[filtered["library_name"].isin(selected_libraries)]
    if selected_licenses:
        filtered = filtered[filtered["license"].isin(selected_licenses)]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_band"] == risk_filter]
    recency_limits = {"≤ 30 days": 30, "≤ 90 days": 90, "≤ 365 days": 365}
    if recency_filter in recency_limits:
        filtered = filtered[filtered["days_since_update"] <= recency_limits[recency_filter]]
    elif recency_filter == "> 365 days":
        filtered = filtered[filtered["days_since_update"] > 365]
    if search_query.strip():
        searchable = filtered[
            ["model_id", "author", "pipeline_tag", "library_name"]
        ].fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered[
            searchable.str.contains(search_query.strip(), case=False, regex=False)
        ]

    if filtered.empty:
        transparent_share = 0.0
    else:
        transparent_share = (
            filtered["has_model_card_like"]
            & ~filtered["license"].str.casefold().isin(UNKNOWN_LICENSES)
        ).mean() * 100
    metric_columns = st.columns(5)
    metric_columns[0].metric("Models loaded", f"{len(models):,}")
    metric_columns[1].metric("Matching filters", f"{len(filtered):,}")
    metric_columns[2].metric(
        "Median downloads", f"{int(filtered['downloads'].median()) if len(filtered) else 0:,}"
    )
    metric_columns[3].metric(
        "Average risk score", f"{filtered['risk_score'].mean():.1f}" if len(filtered) else "0.0"
    )
    metric_columns[4].metric("Transparency coverage", f"{transparent_share:.1f}%")

    if filtered.empty:
        st.info("No models match these filters.")
        return

    buckets = ["0", "1–99", "100–999", "1K–9.9K", "10K–99K", "100K+"]
    bucket_series = pd.cut(
        filtered["downloads"],
        bins=[-1, 0, 99, 999, 9999, 99999, float("inf")],
        labels=buckets,
    )
    bucket_counts = (
        bucket_series.value_counts().reindex(buckets, fill_value=0).rename("models").reset_index()
    )
    bucket_counts.columns = ["bucket", "models"]
    adoption_chart = (
        alt.Chart(bucket_counts)
        .mark_bar(color="#087f73")
        .encode(
            x=alt.X("bucket:N", sort=buckets, title="Downloads"),
            y=alt.Y("models:Q", title="Models"),
            tooltip=["bucket:N", "models:Q"],
        )
        .properties(height=280)
    )
    st.subheader("Adoption distribution")
    st.altair_chart(adoption_chart, width="stretch")

    def top_counts(column):
        return (
            filtered.groupby(column).size().nlargest(10).rename("models").reset_index()
        )

    task_column, license_column = st.columns(2)
    for container, title, column, color in (
        (task_column, "Top tasks", "pipeline_tag", "#087f73"),
        (license_column, "Top licenses", "license", "#bd5b26"),
    ):
        with container:
            st.subheader(title)
            chart = (
                alt.Chart(top_counts(column))
                .mark_bar(color=color)
                .encode(
                    x=alt.X("models:Q", title="Models"),
                    y=alt.Y(f"{column}:N", sort="-x", title=None),
                    tooltip=[f"{column}:N", "models:Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")

    trend_data = (
        filtered.dropna(subset=["lastModified"])
        .assign(
            month=lambda frame: frame["lastModified"]
            .dt.tz_localize(None)
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        .groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "models"})
    )
    st.subheader("Model updates over time")
    if trend_data.empty:
        st.caption("No valid update dates in these records.")
    else:
        trend_chart = (
            alt.Chart(trend_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:T", title="Last modified"),
                y=alt.Y("models:Q", title="Models updated"),
                tooltip=[alt.Tooltip("month:T", title="Month"), alt.Tooltip("models:Q", title="Models")],
            )
            .interactive()
            .properties(height=250)
        )
        st.altair_chart(trend_chart, width="stretch")

    st.subheader("Matching models")
    table = filtered.sort_values("downloads", ascending=False).reset_index(drop=True)
    table_view = table[
        [
            "model_id", "author", "pipeline_tag", "library_name", "license",
            "downloads", "likes", "days_since_update", "risk_score", "risk_band",
        ]
    ].rename(
        columns={
            "model_id": "Model",
            "author": "Author",
            "pipeline_tag": "Task",
            "library_name": "Library",
            "license": "License",
            "downloads": "Downloads",
            "likes": "Likes",
            "days_since_update": "Days since update",
            "risk_score": "Risk score",
            "risk_band": "Risk band",
        }
    )
    table_selection = st.dataframe(
        table_view,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        height=420,
        width="stretch",
        key="model_hub_table",
    )
    st.download_button(
        "Download filtered models as CSV",
        data=table_view.to_csv(index=False),
        file_name="model_hub_filtered.csv",
        mime="text/csv",
    )

    selected_rows = table_selection.selection.rows
    if selected_rows and selected_rows[0] < len(table):
        selected = table.iloc[selected_rows[0]]
        st.subheader(selected["model_id"])
        if selected["model_url"]:
            st.markdown(f"[Open model on {source_label}]({selected['model_url']})")
        detail_columns = st.columns(4)
        detail_columns[0].metric("Downloads", f"{selected['downloads']:,}")
        detail_columns[1].metric("Likes", f"{selected['likes']:,}")
        detail_columns[2].metric("Risk score", f"{selected['risk_score']}")
        detail_columns[3].metric("Risk band", selected["risk_band"])
        st.write(
            f"**Task:** {selected['pipeline_tag']} · **Library:** {selected['library_name']} "
            f"· **License:** {selected['license']} · **Days since update:** {selected['days_since_update']}"
        )
        st.write(
            f"**Model card signal:** {'yes' if selected['has_model_card_like'] else 'no'} "
            f"· **Gated:** {'yes' if selected['gated'] else 'no'} "
            f"· **Disabled:** {'yes' if selected['disabled'] else 'no'}"
        )

# 1. Page Configuration & Title
st.set_page_config(
    page_title="CISA KEV Intelligence",
    page_icon="☁️",
    layout="wide"
)

# 2. Hardcoded Domain Keywords
preset_security = """
Security Guardrails Prompt-Injection Data-Leakage PII-Masking Model-Poisoning 
Adversarial-Attacks Jailbreaking RBAC Encryption Vulnerability-Scanning Anomaly-Detection
Toxicity-Scoring Bias-Detection Privacy-Compliance Model-Defense Firewall OWASP-Top-10
"""

preset_telemetry = """
Telemetry Observability OpenTelemetry Trace-ID Metric-Aggregation Structured-Logging 
Latency Throughput Saturation Context-Propagation Span Distributed-Tracing
Real-Time-Monitoring Logs Metrics Traces Performance-Bottlenecks APM Dashboard
"""

preset_full_llm_ops = """
Token-Consumption TTFT Perplexity Hallucination-Rate Semantic-Similarity ROUGE 
BLEU Cost-Tracking Vector-Database Embedding RAG Cache-Rate Model-Versioning Lineage
Prompt-Engineering Fine-Tuning Inference GPU-Utilization Context-Window Evaluation
"""

# 3. Workspace selection
st.sidebar.header("Workspace")
workspace = st.sidebar.radio("Choose a view", ["CISA dashboard", "Model Hub Intelligence", "Word cloud"])

if workspace == "CISA dashboard":
    render_cisa_dashboard()
elif workspace == "Model Hub Intelligence":
    render_model_hub_dashboard()
else:
    st.title("Interactive AI Operations & Security Word Cloud")
    st.write("Customize keywords, adjust formatting styles, and export high-res images ready for social media.")
    st.sidebar.header("🎨 Styling & Configurations")

    color_theme = st.sidebar.selectbox(
        "Select Color Palette",
        options=["viridis", "plasma", "inferno", "magma", "cool", "spring", "copper", "ocean"]
    )
    bg_color = st.sidebar.selectbox(
        "Background Color",
        options=["black", "white", "darkblue", "charcoal"]
    )
    bg_map = {"black": "black", "white": "white", "darkblue": "#0A192F", "charcoal": "#212529"}
    resolved_bg = bg_map[bg_color]
    dimensions = st.sidebar.radio(
        "Social Media Format Aspect Ratio",
        options=["Standard Web (1200 x 800)", "Square/Instagram (1080 x 1080)", "Banner/X (1500 x 500)"]
    )

    if "Standard" in dimensions:
        w, h = 1200, 800
    elif "Square" in dimensions:
        w, h = 1080, 1080
    else:
        w, h = 1500, 500

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📝 Input Your Text Data")
        preset_choice = st.radio(
            "Load an AI Engineering Vocabulary Preset:",
            options=[
                "Strict AI Security & Guardrails",
                "Telemetry & Observability",
                "Combined Full LLMOps Stack",
                "Clear Canvas (Empty)",
                "Live CISA KEV Catalog",
            ]
        )

        default_text = ""
        input_key = f"wordcloud_input_{preset_choice}"
        if preset_choice == "Strict AI Security & Guardrails":
            default_text = preset_security.strip()
        elif preset_choice == "Telemetry & Observability":
            default_text = preset_telemetry.strip()
        elif preset_choice == "Combined Full LLMOps Stack":
            default_text = preset_security.strip() + " " + preset_telemetry.strip() + " " + preset_full_llm_ops.strip()
        elif preset_choice == "Live CISA KEV Catalog":
            record_limit = st.sidebar.select_slider(
                "Recent CISA vulnerabilities",
                options=[25, 50, 100, 250],
                value=100,
            )
            if st.sidebar.button("Refresh CISA data", use_container_width=True):
                fetch_cisa_kev.clear()

            try:
                with st.spinner("Loading the CISA KEV catalog..."):
                    catalog_version, vulnerabilities = fetch_cisa_kev()
                recent_vulnerabilities = sorted(
                    vulnerabilities,
                    key=lambda item: item.get("dateAdded", ""),
                    reverse=True,
                )[:record_limit]
                text_fields = (
                    "vulnerabilityName",
                    "shortDescription",
                    "vendorProject",
                    "product",
                    "requiredAction",
                )
                default_text = "\n".join(
                    " ".join(str(item[field]) for field in text_fields if item.get(field))
                    for item in recent_vulnerabilities
                )
                input_key = f"wordcloud_input_cisa_{record_limit}_{catalog_version}"
                st.caption(
                    f"Source: [CISA KEV catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) "
                    f"· Version {catalog_version} · {len(recent_vulnerabilities)} recent records · Cached for up to 1 hour."
                )
            except (requests.RequestException, ValueError, AttributeError) as exc:
                st.error(f"Could not load CISA KEV data: {exc}")
                input_key = "wordcloud_input_cisa_unavailable"

        user_input = st.text_area(
            "Edit words or paste custom entries here (separate words with spaces or newlines):",
            value=default_text,
            height=320,
            key=input_key,
        )

    with col2:
        st.subheader("Real-Time Preview Layout")
        if not user_input.strip():
            st.warning("Please type or select a preset on the left panel to populate the layout graphic.")
        else:
            wc = WordCloud(
                width=w,
                height=h,
                background_color=resolved_bg,
                colormap=color_theme,
                collocations=False,
                random_state=42,
            )
            if not wc.process_text(user_input):
                st.warning("Please enter at least one word with two or more characters.")
                st.stop()

            wc.generate(user_input)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig)

            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", bbox_inches="tight", pad_inches=0, dpi=300)
            img_buffer.seek(0)
            plt.close(fig)
            st.download_button(
                label="💾 Download High-Res PNG for Social Media",
                data=img_buffer,
                file_name="ai_ops_wordcloud.png",
                mime="image/png",
                width="stretch"
            )

    st.info("💡 Pro Tip: To make specific terms appear significantly larger on the cloud layout canvas, repeat them multiple times in the text input box panel on the left.")
