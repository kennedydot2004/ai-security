# ai-security

A Streamlit workspace with three views: **CISA dashboard**, **Model Hub Intelligence**, and **Word cloud**.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Hub Intelligence

Summarizes adoption, transparency, and heuristic risk signals for public models, with filters, charts, a sortable table, CSV export, and a per-model detail panel.

### Data source

Currently Hugging Face only, via `GET https://huggingface.co/api/models` with `full=true`, `config=true`, and `limit=2000` (change `limit` in `fetch_hf_models` to sample more or fewer models). Responses are cached for one hour; use the sidebar refresh button to refetch.

### Providers

A provider selector offers `Hugging Face` (active), `OpenML (coming soon)`, and `Kaggle (coming soon)`. The coming-soon options show a placeholder message. Providers are registered in `MODEL_HUB_PROVIDERS`; a new provider supplies its own fetch + normalize step that returns the same normalized columns (including the boolean risk feature columns), and the rest of the view is reused.

### Risk signals and scoring

Normalization produces boolean features; `apply_risk_scores` turns them into a score. The score is a **heuristic**, summed from weighted signals, clipped to 0–100:

| Signal (feature column) | Meaning | Default weight |
| --- | --- | --- |
| `is_unknown_license` | License is unknown or `other` | 25 |
| `is_missing_card` | No model-card-like metadata (`model-index` tag or card data) | 25 |
| `is_stale` | Last modified more than 365 days ago (or unparseable) | 20 |
| `is_gated` | Access is gated | 10 |
| `is_disabled` | Model is disabled | 20 |
| `is_high_adoption_low_transparency` | Downloads in the top 10% and (unknown license or missing card) | 20 |

Bands: **Low** ≤ 29, **Medium** 30–59, **High** ≥ 60.

### Risk-weight controls

The sidebar **Risk weights** expander has one slider per signal (defaults above). Changing a slider recomputes scores and bands immediately, and the metrics, charts, table, and detail panel all use the recomputed values. They exist because the right trade-offs differ by organization (for example, a team that only uses permissive licenses may weight license risk higher), and the defaults are not authoritative.

### Limitations

- The catalog is a sample (`limit=2000` unless changed), not the whole hub.
- Download counts are noisy and can be inflated by automation or mirrors.
- Metadata quality varies; missing fields may reflect incomplete metadata rather than real problems.
- The risk score is a screening heuristic, **not a compliance certification** or security audit.

### Roadmap

OpenML and Kaggle providers are placeholders in the selector and are planned for future iterations.
