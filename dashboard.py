"""Read-only Streamlit view of the vector index benchmark CSVs."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(layout="wide", page_title="Vector Index Benchmark", page_icon="🗄️")

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "benchmark_summary.csv"
RESULTS_PATH = ROOT / "benchmark_results.csv"
INDEXES = {
    "flat": ("Flat (exact)", "#64748B"),
    "hnsw": ("HNSW", "#0D9488"),
    "ivf": ("IVF", "#F59E0B"),
    "pq": ("PQ", "#8B5CF6"),
}
IST = timezone(timedelta(hours=5, minutes=30))


@st.cache_data
def load_data(summary_mtime: int, results_mtime: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Modification times make an updated benchmark invalidate the cache."""
    return pd.read_csv(SUMMARY_PATH), pd.read_csv(RESULTS_PATH)


def name(index_type: str) -> str:
    return INDEXES.get(index_type, (index_type.upper(), "#334155"))[0]


def color(index_type: str) -> str:
    return INDEXES.get(index_type, (index_type.upper(), "#334155"))[1]


def chart_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        legend_title_text="Index",
        margin=dict(l=25, r=25, t=55, b=35),
        height=390,
    )
    return fig


def show_overview(rows: pd.DataFrame, latency_col: str, latency_label: str) -> None:
    if rows.empty:
        st.info("No successful benchmark rows for this corpus size.")
        return

    best_recall = rows["avg_recall_at_k"].max()
    recall_winners = rows.loc[rows["avg_recall_at_k"].eq(best_recall), "index_type"]
    fastest = rows.loc[rows[latency_col].idxmin()]
    smallest = rows.loc[rows["memory_mb"].idxmin()]
    quickest_build = rows.loc[rows["build_time_s"].idxmin()]
    st.subheader("At a glance")
    cards = st.columns(4)
    cards[0].metric("Best recall", f"{best_recall:.2f}")
    cards[0].caption(", ".join(map(name, recall_winners)))
    cards[1].metric(f"Fastest ({latency_label})", f"{fastest[latency_col]:.2f} ms")
    cards[1].caption(name(fastest["index_type"]))
    cards[2].metric("Smallest index", f"{smallest['memory_mb']:.2f} MB")
    cards[2].caption(name(smallest["index_type"]))
    cards[3].metric("Fastest build", f"{quickest_build['build_time_s']:.3f} s")
    cards[3].caption(name(quickest_build["index_type"]))

    st.subheader("Index comparison")
    table = rows[["index_type", "avg_recall_at_k", "p50_latency_ms", "p95_latency_ms", "build_time_s", "memory_mb"]].copy()
    table["index_type"] = table["index_type"].map(name)
    table.columns = ["Index", "Recall@k", "p50 ms", "p95 ms", "Build time s", "Index size MB"]
    st.dataframe(table.round({"Recall@k": 3, "p50 ms": 3, "p95 ms": 3, "Build time s": 3, "Index size MB": 3}), use_container_width=True, hide_index=True)

    scatter = go.Figure()
    max_size = rows["memory_mb"].max()
    for _, row in rows.iterrows():
        index = row["index_type"]
        scatter.add_trace(go.Scatter(
            x=[row[latency_col]], y=[row["avg_recall_at_k"]], mode="markers+text",
            text=[name(index)], textposition="top center", name=name(index),
            marker=dict(color=color(index), size=15 + 35 * row["memory_mb"] / max_size),
            customdata=[[row["memory_mb"]]],
            hovertemplate="%{text}<br>Latency: %{x:.3f} ms<br>Recall@k: %{y:.3f}<br>Index size: %{customdata[0]:.2f} MB<extra></extra>",
        ))
    scatter.update_yaxes(range=[max(0, rows["avg_recall_at_k"].min() - 0.12), 1.08])
    st.plotly_chart(chart_layout(scatter, "Speed vs accuracy trade-off", f"{latency_label} latency (ms)", "Recall@k"), use_container_width=True)
    st.caption("Larger markers represent larger index files on disk.")

    bars = go.Figure()
    ordered = rows.set_index("index_type").reindex([index for index in INDEXES if index in rows["index_type"].values])
    labels = [name(index) for index in ordered.index]
    colors = [color(index) for index in ordered.index]
    bars.add_trace(go.Bar(name="p50", x=labels, y=ordered["p50_latency_ms"], marker_color=colors, opacity=0.55))
    bars.add_trace(go.Bar(name="p95", x=labels, y=ordered["p95_latency_ms"], marker_color=colors))
    chart_layout(bars, "Latency by index", "Index", "Latency (ms)")
    bars.update_layout(barmode="group", legend_title_text="Latency percentile")
    st.plotly_chart(bars, use_container_width=True)

    flat = rows.loc[rows["index_type"].eq("flat")]
    fast_ratio = flat.iloc[0][latency_col] / fastest[latency_col] if not flat.empty and fastest[latency_col] > 0 else None
    size_ratio = flat.iloc[0]["memory_mb"] / smallest["memory_mb"] if not flat.empty and smallest["memory_mb"] > 0 else None
    high_recall = rows.loc[rows["avg_recall_at_k"].ge(0.95), "index_type"]
    takeaways = [
        f"{name(fastest['index_type'])} has the lowest {latency_label} latency" + (f", {fast_ratio:.2f}× faster than {name('flat')}." if fast_ratio else "."),
        f"{name(smallest['index_type'])} has the smallest index" + (f", {size_ratio:.2f}× smaller than {name('flat')}." if size_ratio else "."),
        "Recall@k ≥ 0.95: " + (", ".join(map(name, high_recall)) if not high_recall.empty else "none") + ".",
    ]
    with st.expander("Key takeaways", expanded=True):
        for takeaway in takeaways:
            st.write("• " + takeaway)


def show_scaling(summary: pd.DataFrame, latency_col: str, latency_label: str) -> None:
    if summary["corpus_size"].nunique() < 2:
        st.info("Scaling charts need at least two corpus sizes.")
        return
    log_y = st.checkbox("Log-scale y axis for latency and index size")
    charts = [
        (latency_col, f"{latency_label} latency", "Latency (ms)", log_y),
        ("avg_recall_at_k", "Recall@k", "Recall@k", False),
        ("build_time_s", "Build time", "Seconds", False),
        ("memory_mb", "Index size", "Index size (MB)", log_y),
    ]
    for col, title, unit, y_log in charts:
        fig = go.Figure()
        for index in INDEXES:
            group = summary.loc[summary["index_type"].eq(index)].sort_values("corpus_size")
            if not group.empty:
                fig.add_trace(go.Scatter(x=group["corpus_size"], y=group[col], mode="lines+markers", name=name(index), line_color=color(index)))
        fig = chart_layout(fig, title, "Corpus size (passages, log scale)", unit)
        fig.update_xaxes(type="log", tickvals=sorted(summary["corpus_size"].unique()))
        if y_log:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)


def decode_list(value: str) -> list:
    result = json.loads(value)
    if not isinstance(result, list):
        raise ValueError("Expected a JSON array in benchmark results")
    return result


def show_queries(results: pd.DataFrame, corpus_size: int, k: int) -> None:
    query_rows = results.loc[results["corpus_size"].eq(corpus_size)]
    if query_rows.empty:
        st.info("No successful query results for this corpus size.")
        return
    query = st.selectbox("Query", query_rows["query_text"].drop_duplicates().tolist())
    selected = query_rows.loc[query_rows["query_text"].eq(query)]
    table = selected[["index_type", "latency_ms", "recall_at_k"]].copy()
    table["index_type"] = table["index_type"].map(name)
    table.columns = ["Index", "Latency (ms)", f"Recall@{k}"]
    st.dataframe(table.round(3), use_container_width=True, hide_index=True)
    st.caption("✓ Passage also appears in Flat's exact top-k · ✗ Passage differs from Flat's top-k")
    flat = selected.loc[selected["index_type"].eq("flat")]
    flat_ids = set(map(str, decode_list(flat.iloc[0]["returned_ids"]))) if not flat.empty else set()
    if not flat_ids:
        st.info("Flat results are unavailable, so passage matches cannot be marked.")
    columns = st.columns(4)
    for column, index in zip(columns, INDEXES):
        with column:
            st.subheader(name(index))
            item = selected.loc[selected["index_type"].eq(index)]
            if item.empty:
                st.caption("No result for this index.")
                continue
            ids = decode_list(item.iloc[0]["returned_ids"])
            texts = decode_list(item.iloc[0]["returned_texts"])
            for rank, (passage_id, passage) in enumerate(zip(ids, texts), start=1):
                mark = "✓" if str(passage_id) in flat_ids else "✗"
                passage = str(passage).replace("\n", " ")
                st.markdown(f"**{rank}. {mark}** {passage[:200]}{'…' if len(passage) > 200 else ''}")


def show_methodology(summary: pd.DataFrame, query_count: int) -> None:
    st.graphviz_chart('''digraph {
        rankdir=LR; node [shape=box, style=rounded];
        Data -> "Embeddings\nall-MiniLM-L6-v2 · 384 dimensions" -> "Index layer\nFlat · HNSW · IVF · PQ" -> "Benchmark engine" -> Dashboard;
    }''', use_container_width=True)
    st.subheader("What the measurements mean")
    st.write("**Recall@k** is the fraction of Flat's exact top-k passages that an index retrieved. **p50** is median query latency; **p95** is the latency at or below which 95% of measured searches fall. **Build time** measures index construction. **Index size** is the serialized file size on disk, not RAM usage.")
    st.write("**Flat** uses exact brute-force search. **HNSW** means Hierarchical Navigable Small World, a graph search index. **IVF** means Inverted File, which searches selected vector clusters. **PQ** means Product Quantization, which compresses vectors before search.")
    st.subheader("Parameters used")
    params = summary[["corpus_size", "index_type", "parameters"]].copy()
    params["index_type"] = params["index_type"].map(name)
    params["parameters"] = params["parameters"].map(lambda raw: ", ".join(f"{key}={value}" for key, value in json.loads(raw).items()))
    params.columns = ["Corpus size", "Index", "Parameters"]
    st.dataframe(params, use_container_width=True, hide_index=True)
    st.subheader("Methodology and limitations")
    st.write(f"All indexes use the same embeddings and L2 distance. Flat provides exact top-k ground truth. The benchmark uses {query_count} test queries and corpus sizes up to {summary['corpus_size'].max():,} passages. These measurements came from one machine; absolute latency depends on hardware, so compare ratios and scaling trends. PQ uses nbits=4 where recorded in the parameters; its recall reflects that compression setting. Index size measures disk storage, not RAM.")


def show_selector(summary: pd.DataFrame, corpus_size: int, latency_metric: str) -> None:
    from selector import select_index

    preset = st.selectbox("Preference", ["balanced", "fastest", "accurate", "smallest"])
    min_recall = st.slider("Minimum recall", 0.0, 1.0, 0.0, 0.01)
    max_latency = st.number_input("Maximum latency (ms)", min_value=0.0, value=0.0, help="0 means no limit")
    max_size = st.number_input("Maximum index size (MB)", min_value=0.0, value=0.0, help="0 means no limit")
    result = select_index(summary, corpus_size, preset, min_recall, max_latency or None, max_size or None, latency_metric)
    if not isinstance(result, dict):
        st.write(result)
        return
    winner = result.get("winner")
    if winner is not None:
        st.subheader(f"Recommended: {name(winner) if isinstance(winner, str) else winner}")
    if result.get("explanation"):
        st.write(result["explanation"])
    for key in ("table", "per_index", "scores"):
        if key in result:
            st.dataframe(result[key], use_container_width=True)
            break
    if result.get("weights"):
        st.write("Weights:", result["weights"])


def main() -> None:
    st.title("Benchmarking Vector Indexes for RAG Retrieval")
    st.caption("Explore the speed, recall, build cost, and storage trade-offs of four FAISS indexes.")
    missing = [path.name for path in (SUMMARY_PATH, RESULTS_PATH) if not path.exists()]
    if missing:
        st.error(f"Missing {', '.join(missing)}. Generate benchmark data with `python benchmark.py`.")
        st.stop()

    if st.sidebar.button("Reload data"):
        load_data.clear()
    summary, results = load_data(SUMMARY_PATH.stat().st_mtime_ns, RESULTS_PATH.stat().st_mtime_ns)
    updated = max(SUMMARY_PATH.stat().st_mtime, RESULTS_PATH.stat().st_mtime)
    st.caption("Benchmark last updated: " + datetime.fromtimestamp(updated, IST).strftime("%d %b %Y, %I:%M %p IST"))

    failed_summary = summary.loc[summary["status"].ne("success")]
    failed_results = results.loc[results["status"].ne("success")]
    if not failed_summary.empty or not failed_results.empty:
        details = [f"{row.corpus_size:,} / {name(row.index_type)} ({row.status})" for row in failed_summary.itertuples()]
        details += [f"{row.corpus_size:,} / {name(row.index_type)} / query {row.query_id} ({row.status})" for row in failed_results.itertuples()]
        st.warning("Excluded unsuccessful rows: " + "; ".join(details))
    summary = summary.loc[summary["status"].eq("success")].copy()
    results = results.loc[results["status"].eq("success")].copy()
    if summary.empty:
        st.error("No successful summary rows are available.")
        st.stop()

    sizes = sorted(summary["corpus_size"].unique())
    corpus_size = st.sidebar.selectbox("Corpus size", sizes, index=len(sizes) - 1, format_func=lambda value: f"{value:,} passages")
    latency_label = st.sidebar.radio("Latency metric", ["p50", "p95"], index=1)
    latency_col = f"{latency_label}_latency_ms"
    st.sidebar.caption("All indexes use the same embeddings, same 20 queries, L2 distance; Flat is the exact ground truth for recall.")
    current = summary.loc[summary["corpus_size"].eq(corpus_size)]
    k_values = ", ".join(map(str, sorted(current["k"].unique())))
    repeats = ", ".join(map(str, sorted(current["n_repeats"].unique())))
    warmups = ", ".join(map(str, sorted(current["warmup_searches"].unique())))
    query_count = results.loc[results["corpus_size"].eq(corpus_size), "query_id"].nunique()
    st.caption(f"k = {k_values} · {query_count} queries · {repeats} repeats · {warmups} warm-up searches")

    try:
        from selector import select_index  # noqa: F401
    except ImportError as error:
        if getattr(error, "name", None) != "selector":
            st.warning(f"Index Selector unavailable: {error}")
        has_selector = False
    else:
        has_selector = True
    labels = ["Overview", "Scaling", "Query Explorer", "How it works"]
    if has_selector:
        labels.append("Index Selector")
    tabs = st.tabs(labels)
    renderers = [
        lambda: show_overview(current, latency_col, latency_label),
        lambda: show_scaling(summary, latency_col, latency_label),
        lambda: show_queries(results, corpus_size, int(current["k"].iloc[0])),
        lambda: show_methodology(summary, query_count),
    ]
    if has_selector:
        renderers.append(lambda: show_selector(summary, corpus_size, latency_label))
    for tab, label, render in zip(tabs, labels, renderers):
        with tab:
            try:
                render()
            except Exception as error:
                st.error(f"{label} could not be displayed: {error}")


if __name__ == "__main__":
    main()
