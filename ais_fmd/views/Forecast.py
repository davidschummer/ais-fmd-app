"""Module M14b -- multi-term forecast, with a backtest and budget history."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ais_fmd import auth
from ais_fmd.data import repositories as repo
from ais_fmd.domain import forecast as fc
from ais_fmd.domain.money import format_currency
from ais_fmd.ui import shell, theme

auth.require(auth.Role.TREASURER)

shell.environment_banner()
shell.page_header(
    "Forecast",
    "Next term's dues and committee spending as a range, built from several "
    "past terms and checked against how it would have done before.",
)

bundle = repo.load_bundle()
completed = fc.completed_semesters(bundle.terms)
targets = fc.upcoming_semesters(bundle.terms)

if not completed or not targets:
    shell.empty_state(
        "No completed terms to forecast from",
        "A forecast needs at least one term that has finished.",
    )
    st.stop()

history = fc.term_history(bundle.transactions, bundle.budgets, bundle.terms, completed)

# --- Inputs ------------------------------------------------------------------

default = fc.default_target(bundle.terms)
left, middle, right = st.columns([1, 1, 1])
target = left.selectbox(
    "Forecast for",
    targets,
    index=targets.index(default) if default in targets else 0,
    key="forecast_target",
)
same_season = right.toggle(
    "Same season only",
    value=True,
    key="forecast_same_season",
    help="Forecast Fall from past Falls and Spring from past Springs.",
)

# The slider counts only the terms this forecast could actually use: with
# "Same season only" on, a Spring forecast has the past Springs and nothing
# else, so offering every completed term made the upper settings do nothing.
available, _ = fc.select_basis(history, target, lookback=len(history), same_season=same_season)
if len(available) > 1:
    if st.session_state.get("forecast_lookback", 0) > len(available):
        st.session_state["forecast_lookback"] = len(available)
    lookback = middle.slider(
        "Past terms to use",
        1,
        len(available),
        min(fc.DEFAULT_LOOKBACK, len(available)),
        key="forecast_lookback",
        help="How many of the most recent usable terms the forecast averages.",
    )
else:
    lookback = 1
    middle.caption("Past terms to use: 1 (only one available)")

automatic = fc.forecast(history, target, lookback=lookback, same_season=same_season)

with st.expander("Dues assumptions"):
    st.caption(
        "Filled in from history. Change them to model a new rate or a membership push."
    )
    dues_left, dues_right = st.columns(2)
    dues_rate = dues_left.number_input(
        "Dues rate",
        min_value=0.0,
        value=float(automatic.dues_rate),
        step=5.0,
        format="%.2f",
        key=f"forecast_rate_{target}",
    )
    members = dues_right.number_input(
        "Expected paying members",
        min_value=0,
        value=int(round(automatic.expected_members)),
        step=5,
        key=f"forecast_members_{target}",
    )

result = fc.forecast(
    history, target, lookback=lookback, same_season=same_season,
    dues_rate=dues_rate, expected_members=members,
)
tested = fc.backtest(history, lookback=lookback, same_season=same_season)

if result.fell_back_to_all_seasons:
    shell.notify("info", f"No past {fc.season_of(target)} term yet, so every season was used.")
if result.is_thin:
    shell.notify(
        "warning",
        "Only one past term behind this forecast, so the range is a rule of thumb "
        "(±25%), not measured.",
    )

# --- Headline ----------------------------------------------------------------

st.markdown('<hr class="ais-rule" />', unsafe_allow_html=True)
net = result.line("Net")
income = result.line("Total income")
expenses = result.line("Total expenses")



def dollars(value: float) -> str:
    """Short enough for a tile on a phone: $5,145, or -$10.7k past four digits."""
    sign = "-" if value < 0 else ""
    size = abs(value)
    return f"{sign}${size / 1000:,.1f}k" if size >= 10_000 else f"{sign}${size:,.0f}"


shell.metric_row([
    {"label": "Income", "value": dollars(income.expected), "currency": False},
    {"label": "Expenses", "value": dollars(expenses.expected), "currency": False},
    {"label": "Net", "value": dollars(net.expected), "currency": False},
    {
        "label": "Avg forecast error",
        "value": "—" if tested.is_empty else dollars(tested.mean_absolute_error),
        "currency": False,
    },
])
shell.say(
    f"Using {', '.join(result.basis)} (recent terms weighted more). "
    f"Likely net for {target}: {format_currency(net.low)} to {format_currency(net.high)}. "
    f"Avg forecast error is how far a line item was off, on average, when past "
    f"terms were forecast this way.",
    caption=True,
)

# --- Forecast ----------------------------------------------------------------

palette = theme.active()
table_column, chart_column = st.columns([1, 1])
money = st.column_config.NumberColumn(format="dollar")

with table_column:
    st.markdown("**Low / expected / high**")
    shell.dataframe(
        result.frame().drop(columns=["Kind"]),
        column_config={"Low": money, "Expected": money, "High": money},
    )

with chart_column:
    st.markdown("**Committee spending range**")
    committees = list(reversed(result.committees()))
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=[c.line for c in committees],
            x=[c.high - c.low for c in committees],
            base=[c.low for c in committees],
            orientation="h",
            marker=dict(color=palette.accent_wash, line=dict(color=palette.accent, width=1)),
            name="Likely range",
            hovertemplate="%{y}: $%{base:,.0f} to $%{x:,.0f} wide<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            y=[c.line for c in committees],
            x=[c.expected for c in committees],
            mode="markers",
            marker=dict(color=palette.ink, size=9, symbol="line-ns-open", line=dict(width=3)),
            name="Expected",
            hovertemplate="%{y}: $%{x:,.2f} expected<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(260, 34 * len(committees) + 80),
        xaxis=dict(tickprefix="$", separatethousands=True, title=None),
        yaxis=dict(title=None),
        showlegend=False,
    )
    shell.chart(figure, key="forecast_ranges")

# --- Backtest ----------------------------------------------------------------

st.markdown('<hr class="ais-rule" />', unsafe_allow_html=True)
st.markdown("#### How accurate would this have been?")
if tested.is_empty:
    st.caption("Not enough history to backtest: every term needs at least one term before it.")
else:
    shell.say(
        f"Each past term forecast from only the terms before it, with these settings. "
        f"Avg forecast error: {format_currency(tested.mean_absolute_error)} per line item; "
        f"{tested.share_in_range:.0f}% of actual figures landed inside the range.",
        caption=True,
    )
    net_rows = tested.detail[tested.detail["Line"] == "Net"]
    backtest_chart = go.Figure()
    backtest_chart.add_trace(
        go.Scatter(
            x=net_rows["Semester"],
            y=net_rows["Expected"],
            error_y=dict(
                type="data", symmetric=False,
                array=net_rows["High"] - net_rows["Expected"],
                arrayminus=net_rows["Expected"] - net_rows["Low"],
                color=palette.accent,
            ),
            mode="markers",
            marker=dict(color=palette.accent, size=10),
            name="Forecast net (range)",
            hovertemplate="%{x} forecast: $%{y:,.2f}<extra></extra>",
        )
    )
    backtest_chart.add_trace(
        go.Scatter(
            x=net_rows["Semester"],
            y=net_rows["Actual"],
            mode="markers",
            marker=dict(color=palette.ink, size=10, symbol="diamond"),
            name="Actual net",
            hovertemplate="%{x} actual: $%{y:,.2f}<extra></extra>",
        )
    )
    backtest_chart.update_layout(
        height=300,
        yaxis=dict(tickprefix="$", separatethousands=True, title="Net"),
        xaxis=dict(title=None),
    )
    shell.chart(backtest_chart, key="forecast_backtest")
    shell.dataframe(
        tested.summary,
        column_config={
            "MAE": st.column_config.NumberColumn("Avg forecast error", format="dollar"),
            "Net forecast": money,
            "Net actual": money,
            "Net error": money,
            "% lines in range": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )

# --- Budget history ----------------------------------------------------------

st.markdown('<hr class="ais-rule" />', unsafe_allow_html=True)
st.markdown("#### Budget used, by committee")
st.caption(
    "Share of each committee's budget it spent, most recent terms on the right. "
    "Blank means no budget was set that term."
)
utilization = fc.budget_utilization(history)
if utilization.empty:
    st.caption("No budgets recorded yet.")
else:
    percent = st.column_config.NumberColumn(format="%.0f%%")
    shell.dataframe(
        utilization,
        column_config={column: percent for column in utilization.columns if column != "Committee"},
    )

# --- Export ------------------------------------------------------------------

st.markdown('<hr class="ais-rule" />', unsafe_allow_html=True)
st.markdown("#### Executive summary")
export_left, export_right = st.columns(2)
slug = target.lower().replace(" ", "-")
export_left.download_button(
    "Download summary (Markdown)",
    data=fc.executive_summary(result, tested),
    file_name=f"forecast-{slug}.md",
    mime="text/markdown",
    key="forecast_summary_download",
)
export_right.download_button(
    "Download forecast table (CSV)",
    data=result.frame().to_csv(index=False),
    file_name=f"forecast-{slug}.csv",
    mime="text/csv",
    key="forecast_csv_download",
)
st.caption(
    "The summary is ready to paste into a board update, or to hand to Claude to "
    "write up. A forecast is a planning aid: it assumes next term resembles the "
    "terms it was built from."
)
