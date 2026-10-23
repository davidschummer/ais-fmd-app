"""
Module M14b -- forecasting a term from several past terms.

The scenario planner (M14) projects from *one* baseline term. One term is a
noisy thing to set a budget from: a single big trip or a missing sponsor moves
everything. This module forecasts each line -- dues, other income, every
committee's spend -- from several completed terms, as a low / expected / high
range, and backtests itself so the range comes with a track record.

The method is deliberately plain, so a treasurer can check it by hand:

* **Same season.** Fall is forecast from past Falls and Spring from past
  Springs -- the dues drive is in August and Formal is in spring, so the two
  seasons do not look alike. If there is no past term of that season, every
  completed term is used instead, and the forecast says so.
* **Recency-weighted average.** The most recent term counts fully; each term
  before it counts `RECENCY_DECAY` times as much as the one after it.
* **Range = expected +/- one weighted standard deviation** of the history,
  never narrower than `MIN_SPREAD_SHARE` of the expected value. With fewer than
  two terms there is no spread to measure, so `THIN_HISTORY_SPREAD_SHARE` is
  used and the forecast is flagged as thin.
* **Dues = paying members x rate.** Members are counted by payments, because a
  bank memo does not always carry a readable name; the rate is the average
  actually paid. Both can be overridden for "what if dues go to $50".

Only completed terms are used. A term still in progress would read as a term
with half its spending, and drag every forecast down.

Pure pandas -- no Streamlit -- so it runs unchanged on the real history.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from ..config.categories import BUDGETED_COMMITTEE_IDS, committee_name
from . import budgets as budget_domain
from . import dues as dues_domain
from .money import format_currency, safe_percent
from .terms import attach_semester, prepare_terms

RECENCY_DECAY = 0.75
MIN_SPREAD_SHARE = 0.10
THIN_HISTORY_SPREAD_SHARE = 0.25
DEFAULT_LOOKBACK = 4
DEFAULT_UTILIZATION_TERMS = 8

_SEASON_ORDER = ("Spring", "Fall")
_NAME_RE = re.compile(r"^\s*(fall|spring|summer|winter)\s+((?:19|20)\d{2})\s*$", re.IGNORECASE)


def season_of(semester: str) -> str:
    """'Fall 2025' -> 'Fall'. Empty for a name without a season."""
    parts = str(semester or "").split()
    return parts[0].title() if parts else ""


def _is_summer(semester: str) -> bool:
    return season_of(semester) == "Summer"


# --- History -------------------------------------------------------------------


@dataclass(frozen=True)
class TermActuals:
    """What actually happened in one completed term."""

    semester: str
    dues_income: float
    dues_payments: int
    other_income: float
    committee_spend: dict[int, float] = field(default_factory=dict)
    other_expenses: float = 0.0
    budgets: dict[int, float] = field(default_factory=dict)

    @property
    def season(self) -> str:
        return season_of(self.semester)

    @property
    def income(self) -> float:
        return self.dues_income + self.other_income

    @property
    def committee_total(self) -> float:
        return sum(self.committee_spend.values())

    @property
    def expenses(self) -> float:
        return self.committee_total + self.other_expenses

    @property
    def net(self) -> float:
        return self.income - self.expenses

    @property
    def is_empty(self) -> bool:
        return self.income == 0 and self.expenses == 0


def completed_semesters(df_terms: pd.DataFrame, as_of: pd.Timestamp | None = None) -> list[str]:
    """Terms that ended before `as_of`, oldest first. Summer terms are left out."""
    terms = prepare_terms(df_terms)
    if terms.empty:
        return []
    cutoff = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    done = terms[terms["end_date"] < cutoff]
    return [s for s in done["Semester"].dropna().astype(str) if not _is_summer(s)]


def term_history(
    df_transactions: pd.DataFrame,
    df_budgets: pd.DataFrame,
    df_terms: pd.DataFrame,
    semesters: list[str],
) -> list[TermActuals]:
    """
    Actuals for each named term, in the order given.

    Tags the transactions once rather than once per term -- the real history
    reaches back to 2021, and a backtest asks for every term of it.
    """
    totals = budget_domain.semester_totals_by_semester(df_transactions, df_terms, semesters)
    dues = dues_domain.compare_semesters(df_transactions, df_terms, semesters).set_index("Semester")

    spend: dict[tuple[str, int], float] = {}
    tagged = attach_semester(df_transactions, df_terms)
    if not tagged.empty:
        expenses = tagged[
            (tagged["amount"] < 0) & tagged["budget_category"].isin(BUDGETED_COMMITTEE_IDS)
        ]
        if not expenses.empty:
            grouped = expenses["amount"].abs().groupby(
                [expenses["Semester"], expenses["budget_category"].astype(int)]
            ).sum()
            spend = {(str(s), int(c)): float(v) for (s, c), v in grouped.items()}

    budget_by_term: dict[tuple[str, int], float] = {}
    if not df_budgets.empty and not df_terms.empty:
        term_ids = df_terms.drop_duplicates(subset=["Semester"]).set_index("Semester")["TermID"]
        rows = df_budgets[df_budgets["committeeid"].isin(BUDGETED_COMMITTEE_IDS)]
        per_term = rows.groupby(["termid", "committeeid"])["budget_amount"].sum()
        for semester in semesters:
            term_id = term_ids.get(semester)
            if term_id is None:
                continue
            for (tid, cid), amount in per_term.items():
                if tid == term_id:
                    budget_by_term[(semester, int(cid))] = float(amount)

    history = []
    for semester in semesters:
        income = totals[semester]["income"]
        expenses = totals[semester]["expenses"]
        dues_income = float(dues.loc[semester, "Collected"]) if semester in dues.index else 0.0
        payments = int(dues.loc[semester, "Payments"]) if semester in dues.index else 0
        committee_spend = {
            cid: spend[(semester, cid)] for cid in BUDGETED_COMMITTEE_IDS if (semester, cid) in spend
        }
        history.append(
            TermActuals(
                semester=semester,
                dues_income=dues_income,
                dues_payments=payments,
                # Dues reach back before the term starts (dues_terms), so they can
                # exceed the income booked inside the term's dates; never negative.
                other_income=max(income - dues_income, 0.0),
                committee_spend=committee_spend,
                other_expenses=max(expenses - sum(committee_spend.values()), 0.0),
                budgets={
                    cid: budget_by_term[(semester, cid)]
                    for cid in BUDGETED_COMMITTEE_IDS
                    if (semester, cid) in budget_by_term
                },
            )
        )
    return history


# --- Forecast ------------------------------------------------------------------


@dataclass(frozen=True)
class LineForecast:
    line: str
    kind: str  # "income", "expense" or "net"
    low: float
    expected: float
    high: float
    committee_id: int | None = None


@dataclass
class Forecast:
    target: str
    basis: list[str]
    weights: list[float]
    lines: list[LineForecast]
    expected_members: float
    dues_rate: float
    fell_back_to_all_seasons: bool = False

    @property
    def is_thin(self) -> bool:
        """Fewer than two terms behind it: the range is a rule of thumb, not measured."""
        return len(self.basis) < 2

    @property
    def is_empty(self) -> bool:
        return not self.basis

    def line(self, name: str) -> LineForecast:
        return next(item for item in self.lines if item.line == name)

    def committees(self) -> list[LineForecast]:
        return [item for item in self.lines if item.committee_id is not None]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Line": item.line, "Kind": item.kind, "Low": item.low,
                 "Expected": item.expected, "High": item.high}
                for item in self.lines
            ],
            columns=["Line", "Kind", "Low", "Expected", "High"],
        )


def recency_weights(count: int, decay: float = RECENCY_DECAY) -> list[float]:
    """Weights for `count` terms, oldest first, summing to 1."""
    if count <= 0:
        return []
    raw = [decay ** (count - 1 - i) for i in range(count)]
    total = sum(raw)
    return [w / total for w in raw]


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    return sum(v * w for v, w in zip(values, weights)) if values else 0.0


def _weighted_std(values: list[float], weights: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _weighted_mean(values, weights)
    return math.sqrt(sum(w * (v - mean) ** 2 for v, w in zip(values, weights)))


def _spread(values: list[float], weights: list[float], floor_base: float) -> float:
    share = THIN_HISTORY_SPREAD_SHARE if len(values) < 2 else MIN_SPREAD_SHARE
    return max(_weighted_std(values, weights), share * abs(floor_base))


def _line(name: str, kind: str, expected: float, spread: float, committee_id=None,
          *, floor_at_zero: bool = True) -> LineForecast:
    low = expected - spread
    if floor_at_zero:
        low = max(low, 0.0)
    return LineForecast(name, kind, round(low, 2), round(expected, 2), round(expected + spread, 2),
                        committee_id)


def select_basis(
    history: list[TermActuals],
    target: str,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    same_season: bool = True,
) -> tuple[list[TermActuals], bool]:
    """
    The terms a forecast of `target` rests on, oldest first, and whether it had
    to fall back from same-season to every season.

    `history` must already be limited to terms before the target.
    """
    usable = [term for term in history if not term.is_empty and term.semester != target]
    fell_back = False
    if same_season:
        matching = [term for term in usable if term.season == season_of(target)]
        if matching:
            usable = matching
        elif usable:
            fell_back = True
    return usable[-max(lookback, 1):], fell_back


def forecast(
    history: list[TermActuals],
    target: str,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    same_season: bool = True,
    dues_rate: float | None = None,
    expected_members: float | None = None,
) -> Forecast:
    """Low / expected / high for every line of `target`, from the terms in `history`."""
    basis, fell_back = select_basis(history, target, lookback=lookback, same_season=same_season)
    weights = recency_weights(len(basis))

    def series(pick) -> list[float]:
        return [float(pick(term)) for term in basis]

    # Dues: members x rate.
    members_hist = series(lambda t: t.dues_payments)
    members_mean = _weighted_mean(members_hist, weights)
    dues_hist = series(lambda t: t.dues_income)
    paid_per_member = (
        _weighted_mean(dues_hist, weights) / members_mean if members_mean else 0.0
    )
    rate = float(dues_rate) if dues_rate is not None else paid_per_member
    members = float(expected_members) if expected_members is not None else members_mean
    member_spread = _spread(members_hist, weights, members_mean)
    member_share = member_spread / members_mean if members_mean else THIN_HISTORY_SPREAD_SHARE
    dues_line = _line("Dues income", "income", members * rate, members * member_share * rate)

    other_income_hist = series(lambda t: t.other_income)
    other_income_mean = _weighted_mean(other_income_hist, weights)
    other_income_line = _line(
        "Other income", "income", other_income_mean,
        _spread(other_income_hist, weights, other_income_mean),
    )

    committee_lines = []
    active = [cid for cid in BUDGETED_COMMITTEE_IDS if any(t.committee_spend.get(cid) for t in basis)]
    for cid in active:
        values = series(lambda t, cid=cid: t.committee_spend.get(cid, 0.0))
        mean = _weighted_mean(values, weights)
        committee_lines.append(
            _line(committee_name(cid), "expense", mean, _spread(values, weights, mean), cid)
        )
    committee_lines.sort(key=lambda item: -item.expected)

    other_exp_hist = series(lambda t: t.other_expenses)
    other_exp_mean = _weighted_mean(other_exp_hist, weights)
    other_exp_line = _line(
        "Other expenses", "expense", other_exp_mean,
        _spread(other_exp_hist, weights, other_exp_mean),
    )

    # Totals are sums of the expected lines, so the table adds up; their ranges
    # come from how the totals themselves moved, which is narrower than adding
    # every line's worst case together (lines partly offset each other).
    income_expected = dues_line.expected + other_income_line.expected
    expense_expected = sum(item.expected for item in committee_lines) + other_exp_line.expected
    income_line = _line(
        "Total income", "income", income_expected,
        _spread(series(lambda t: t.income), weights, income_expected),
    )
    expense_line = _line(
        "Total expenses", "expense", expense_expected,
        _spread(series(lambda t: t.expenses), weights, expense_expected),
    )
    net_line = _line(
        "Net", "net", income_expected - expense_expected,
        _spread(series(lambda t: t.net), weights, expense_expected),
        floor_at_zero=False,
    )

    lines = [dues_line, other_income_line, income_line, *committee_lines, other_exp_line,
             expense_line, net_line]
    if not basis:
        lines = [LineForecast(item.line, item.kind, 0.0, 0.0, 0.0, item.committee_id) for item in lines]

    return Forecast(
        target=target,
        basis=[term.semester for term in basis],
        weights=[round(w, 3) for w in weights],
        lines=lines,
        expected_members=round(members, 1),
        dues_rate=round(rate, 2),
        fell_back_to_all_seasons=fell_back,
    )


# --- Backtest ------------------------------------------------------------------


@dataclass
class Backtest:
    summary: pd.DataFrame
    detail: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        return self.summary.empty

    @property
    def mean_absolute_error(self) -> float | None:
        return None if self.summary.empty else float(self.summary["MAE"].mean())

    @property
    def share_in_range(self) -> float | None:
        if self.detail.empty:
            return None
        return float(self.detail["In range"].mean() * 100)


_SUMMARY_COLUMNS = ["Semester", "Terms used", "Same season", "MAE", "Net forecast",
                    "Net actual", "Net error", "% lines in range"]
_DETAIL_COLUMNS = ["Semester", "Line", "Low", "Expected", "High", "Actual", "Error", "In range"]


def _actual_lines(term: TermActuals) -> dict[str, float]:
    lines = {
        "Dues income": term.dues_income,
        "Other income": term.other_income,
        "Other expenses": term.other_expenses,
    }
    for cid, amount in term.committee_spend.items():
        lines[committee_name(cid)] = amount
    return lines


def backtest(
    history: list[TermActuals],
    *,
    lookback: int = DEFAULT_LOOKBACK,
    same_season: bool = True,
) -> Backtest:
    """
    Forecast each completed term from only the terms before it, and score it.

    MAE is the mean absolute error across the term's line items (dues, other
    income, each committee, other expenses) -- in dollars, so it reads as "a
    typical line was off by this much". A term with nothing before it is
    skipped: there is nothing to forecast it from.
    """
    summary_rows, detail_rows = [], []
    for index, term in enumerate(history):
        if term.is_empty:
            continue
        result = forecast(history[:index], term.semester, lookback=lookback, same_season=same_season)
        if result.is_empty:
            continue
        actual = _actual_lines(term)
        predicted = {item.line: item for item in result.lines
                     if item.line in ("Dues income", "Other income", "Other expenses")
                     or item.committee_id is not None}
        errors, inside = [], []
        for name in sorted(set(actual) | set(predicted)):
            item = predicted.get(name, LineForecast(name, "expense", 0.0, 0.0, 0.0))
            value = actual.get(name, 0.0)
            error = item.expected - value
            in_range = item.low <= value <= item.high
            errors.append(abs(error))
            inside.append(in_range)
            detail_rows.append({
                "Semester": term.semester, "Line": name, "Low": item.low,
                "Expected": item.expected, "High": item.high, "Actual": round(value, 2),
                "Error": round(error, 2), "In range": in_range,
            })
        net = result.line("Net")
        net_in_range = net.low <= term.net <= net.high
        inside.append(net_in_range)
        detail_rows.append({
            "Semester": term.semester, "Line": "Net", "Low": net.low, "Expected": net.expected,
            "High": net.high, "Actual": round(term.net, 2),
            "Error": round(net.expected - term.net, 2), "In range": net_in_range,
        })
        summary_rows.append({
            "Semester": term.semester,
            "Terms used": ", ".join(result.basis),
            "Same season": not result.fell_back_to_all_seasons,
            "MAE": round(sum(errors) / len(errors), 2) if errors else 0.0,
            "Net forecast": net.expected,
            "Net actual": round(term.net, 2),
            "Net error": round(net.expected - term.net, 2),
            "% lines in range": round(100 * sum(inside) / len(inside), 1),
        })
    return Backtest(
        summary=pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS),
        detail=pd.DataFrame(detail_rows, columns=_DETAIL_COLUMNS),
    )


# --- Budget utilization --------------------------------------------------------


def budget_utilization(
    history: list[TermActuals],
    *,
    terms: int = DEFAULT_UTILIZATION_TERMS,
) -> pd.DataFrame:
    """
    % of budget each committee spent, one column per recent term, plus a
    recency-weighted average. A term where the committee had no budget is
    blank rather than infinite (FINDING F12).
    """
    recent = [term for term in history if not term.is_empty][-max(terms, 1):]
    semesters = [term.semester for term in recent]
    columns = ["Committee", *semesters, "Weighted avg"]
    rows = []
    for cid in BUDGETED_COMMITTEE_IDS:
        if not any(t.budgets.get(cid) or t.committee_spend.get(cid) for t in recent):
            continue
        row: dict[str, object] = {"Committee": committee_name(cid)}
        values, kept = [], []
        for position, term in enumerate(recent):
            percent = safe_percent(term.committee_spend.get(cid, 0.0), term.budgets.get(cid, 0.0))
            row[term.semester] = None if percent is None else round(percent, 1)
            if percent is not None:
                values.append(percent)
                kept.append(position)
        if values:
            # Weight by how recent each term is among *all* the terms shown,
            # so a gap year does not make an old term count as the latest.
            raw = [RECENCY_DECAY ** (len(recent) - 1 - p) for p in kept]
            row["Weighted avg"] = round(sum(v * w for v, w in zip(values, raw)) / sum(raw), 1)
        else:
            row["Weighted avg"] = None
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


# --- Choosing a target ---------------------------------------------------------


def _next_name(semester: str) -> str | None:
    match = _NAME_RE.match(str(semester))
    if not match:
        return None
    season, year = match.group(1).title(), int(match.group(2))
    if season in ("Fall", "Winter"):
        return f"Spring {year + 1}"
    return f"Fall {year}"  # Spring and Summer both lead to Fall


def upcoming_semesters(
    df_terms: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    *,
    extra: int = 2,
) -> list[str]:
    """
    Terms worth forecasting: defined terms that have not finished yet, then
    `extra` more by name, since next year's terms are rarely set up in advance.
    """
    terms = prepare_terms(df_terms)
    cutoff = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    names = []
    if not terms.empty:
        open_terms = terms[terms["end_date"] >= cutoff]
        names = [s for s in open_terms["Semester"].astype(str) if not _is_summer(s)]
    last = str(terms["Semester"].iloc[-1]) if not terms.empty else None
    for _ in range(extra):
        following = _next_name(last) if last else None
        if following is None:
            break
        names.append(following)
        last = following
    return names


def default_target(df_terms: pd.DataFrame, as_of: pd.Timestamp | None = None) -> str | None:
    """The next term that has not started yet -- the one a budget is being set for."""
    terms = prepare_terms(df_terms)
    cutoff = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    upcoming = upcoming_semesters(df_terms, cutoff)
    started = set(terms[terms["start_date"] <= cutoff]["Semester"].astype(str)) if not terms.empty else set()
    for name in upcoming:
        if name not in started:
            return name
    return upcoming[0] if upcoming else None


# --- Executive summary ---------------------------------------------------------


def _money_range(item: LineForecast) -> str:
    return f"{format_currency(item.low)} to {format_currency(item.high)}"


def executive_summary(result: Forecast, tested: Backtest) -> str:
    """
    A board-ready summary in Markdown, for download. Not for st.markdown:
    it is full of dollar signs (HANDOFF.md trap 1).
    """
    if result.is_empty:
        return f"# {result.target} forecast\n\nNo completed terms to forecast from.\n"

    net = result.line("Net")
    income = result.line("Total income")
    expenses = result.line("Total expenses")
    dues = result.line("Dues income")
    outcome = "surplus" if net.expected >= 0 else "deficit"

    lines = [
        f"# {result.target} budget forecast",
        "",
        "## Headline",
        "",
        f"- Expected net: **{format_currency(net.expected)}** ({outcome}); "
        f"likely range {_money_range(net)}.",
        f"- Income: {format_currency(income.expected)} expected ({_money_range(income)}).",
        f"- Expenses: {format_currency(expenses.expected)} expected ({_money_range(expenses)}).",
        f"- Dues: about {result.expected_members:.0f} paying members at "
        f"{format_currency(result.dues_rate)} = {format_currency(dues.expected)}.",
        "",
        "## Committee spending",
        "",
        "| Committee | Low | Expected | High |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in result.committees():
        lines.append(
            f"| {item.line} | {format_currency(item.low)} | "
            f"{format_currency(item.expected)} | {format_currency(item.high)} |"
        )

    lines += ["", "## How accurate has this been?", ""]
    if tested.is_empty:
        lines.append("Not enough history to backtest yet.")
    else:
        lines.append(
            f"Forecasting each past term from the terms before it, the average forecast error "
            f"was {format_currency(tested.mean_absolute_error)} per line item, and "
            f"{tested.share_in_range:.0f}% of actual figures landed inside the forecast range."
        )
        lines += ["", "| Term | Avg forecast error | Net forecast | Net actual |",
                  "| --- | ---: | ---: | ---: |"]
        for row in tested.summary.itertuples(index=False):
            lines.append(
                f"| {row.Semester} | {format_currency(row.MAE)} | "
                f"{format_currency(row[4])} | {format_currency(row[5])} |"
            )

    basis = ", ".join(result.basis)
    lines += [
        "",
        "## Method",
        "",
        f"Based on {basis}, with recent terms weighted more heavily. The range is one "
        f"standard deviation of past results either side of the expected figure.",
    ]
    if result.fell_back_to_all_seasons:
        lines.append(f"No past {season_of(result.target)} term was available, so all seasons were used.")
    if result.is_thin:
        lines.append("Only one past term was available, so the range is a rule of thumb (+/-25%).")
    return "\n".join(lines) + "\n"
