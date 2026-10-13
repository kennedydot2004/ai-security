import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import requests
import altair as alt
import pandas as pd

CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cisa_kev():
    response = requests.get(CISA_KEV_URL, timeout=15)
    response.raise_for_status()
    catalog = response.json()
    vulnerabilities = catalog.get("vulnerabilities") if isinstance(catalog, dict) else None
    if not isinstance(vulnerabilities, list):
        raise ValueError("CISA returned an unexpected catalog format.")
    return catalog.get("catalogVersion", "unknown"), vulnerabilities


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
workspace = st.sidebar.radio("Choose a view", ["CISA dashboard", "Word cloud"])

if workspace == "CISA dashboard":
    render_cisa_dashboard()
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
