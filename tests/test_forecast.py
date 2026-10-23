"""Tests for multi-term forecasting (M14b)."""

from __future__ import annotations

import pandas as pd
import pytest

from ais_fmd.domain import forecast as fc

MEETING_FOOD, MERCH = 8, 13

TERMS = pd.DataFrame(
    {
        "TermID": ["FA24", "SP25", "SU25", "FA25", "SP26", "FA26"],
        "Semester": ["Fall 2024", "Spring 2025", "Summer 2025", "Fall 2025", "Spring 2026", "Fall 2026"],
        "start_date": ["2024-08-21", "2025-01-06", "2025-05-10", "2025-08-20", "2026-01-05", "2026-08-15"],
        "end_date": ["2024-12-13", "2025-05-02", "2025-08-01", "2025-12-12", "2026-05-01", "2026-12-18"],
    }
)
AS_OF = pd.Timestamp("2026-10-16")


def _rows(day: str, dues: int, food: float, merch: float = 0.0) -> list[dict]:
    rows = [
        {"transaction_date": day, "amount": 40.0, "budget_category": 1,
         "details": f"ZELLE FROM MEMBER {i} ON 09/01 REF # {day}{i} DUES"}
        for i in range(dues)
    ]
    rows.append({"transaction_date": day, "amount": -food, "budget_category": MEETING_FOOD,
                 "details": "PURCHASE AUTHORIZED PUBLIX"})
    if merch:
        rows.append({"transaction_date": day, "amount": -merch, "budget_category": MERCH,
                     "details": "PURCHASE AUTHORIZED VISTAPRINT"})
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["budget_category"] = pd.array(df["budget_category"], dtype="Int64")
    df["account"] = "Wells Fargo"
    df["purpose"] = ""
    return df


@pytest.fixture
def ledger():
    transactions = _frame(
        _rows("2024-09-10", dues=10, food=1000.0, merch=200.0)
        + _rows("2025-02-10", dues=8, food=600.0)
        + _rows("2025-09-10", dues=12, food=1400.0, merch=400.0)
        + _rows("2026-02-10", dues=6, food=800.0)
        + _rows("2026-09-10", dues=20, food=50.0)  # in progress on AS_OF
    )
    budgets = pd.DataFrame(
        {"termid": ["FA24", "FA25", "FA25"], "committeeid": [MEETING_FOOD, MEETING_FOOD, MERCH],
         "budget_amount": [2000.0, 1000.0, 0.0]}
    )
    return transactions, budgets


@pytest.fixture
def history(ledger):
    transactions, budgets = ledger
    return fc.term_history(transactions, budgets, TERMS, fc.completed_semesters(TERMS, AS_OF))


# --- History -------------------------------------------------------------------


def test_only_finished_non_summer_terms_count_as_history():
    assert fc.completed_semesters(TERMS, AS_OF) == [
        "Fall 2024", "Spring 2025", "Fall 2025", "Spring 2026",
    ]


def test_history_reflects_each_terms_actuals(history):
    fall_2025 = next(t for t in history if t.semester == "Fall 2025")
    assert fall_2025.dues_income == pytest.approx(480.0)
    assert fall_2025.dues_payments == 12
    assert fall_2025.committee_spend == {MEETING_FOOD: 1400.0, MERCH: 400.0}
    assert fall_2025.net == pytest.approx(480.0 - 1800.0)


def test_recency_weights_favour_recent_terms():
    weights = fc.recency_weights(3)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]
    assert fc.recency_weights(0) == []


# --- Forecast ------------------------------------------------------------------


def test_forecast_uses_same_season_terms_weighted_toward_recent(history):
    result = fc.forecast(history, "Fall 2026")
    assert result.basis == ["Fall 2024", "Fall 2025"]
    old, new = fc.recency_weights(2)
    food = result.line("Meeting Food")
    assert food.expected == pytest.approx(old * 1000 + new * 1400, abs=0.01)
    assert food.low <= food.expected <= food.high


def test_every_line_reads_low_expected_high(history):
    for item in fc.forecast(history, "Spring 2027").lines:
        assert item.low <= item.expected <= item.high, item.line
        if item.kind != "net":
            assert item.low >= 0, item.line


def test_totals_add_up_from_the_lines(history):
    result = fc.forecast(history, "Fall 2026")
    expenses = sum(i.expected for i in result.committees()) + result.line("Other expenses").expected
    assert result.line("Total expenses").expected == pytest.approx(expenses, abs=0.02)
    net = result.line("Total income").expected - result.line("Total expenses").expected
    assert result.line("Net").expected == pytest.approx(net, abs=0.02)


def test_dues_are_members_times_rate_and_can_be_overridden(history):
    result = fc.forecast(history, "Fall 2026", dues_rate=50.0, expected_members=30)
    assert result.line("Dues income").expected == pytest.approx(1500.0)
    baseline = fc.forecast(history, "Fall 2026")
    assert result.line("Net").expected > baseline.line("Net").expected


def test_lookback_limits_the_terms_used(history):
    assert fc.forecast(history, "Fall 2026", lookback=1).basis == ["Fall 2025"]


def test_falls_back_to_all_seasons_when_no_same_season_term(history):
    result = fc.forecast(history[:1], "Spring 2025")
    assert result.basis == ["Fall 2024"]
    assert result.fell_back_to_all_seasons


def test_mixing_seasons_on_request(history):
    result = fc.forecast(history, "Fall 2026", same_season=False, lookback=8)
    assert result.basis == ["Fall 2024", "Spring 2025", "Fall 2025", "Spring 2026"]


def test_thin_history_gets_a_rule_of_thumb_range(history):
    result = fc.forecast(history[:1], "Fall 2025")
    food = result.line("Meeting Food")
    assert result.is_thin
    assert food.high - food.expected == pytest.approx(0.25 * 1000)


def test_no_history_gives_an_empty_forecast_not_an_error():
    result = fc.forecast([], "Fall 2026")
    assert result.is_empty
    assert all(item.expected == 0 for item in result.lines)
    assert "No completed terms" in fc.executive_summary(result, fc.backtest([]))


def test_an_empty_term_is_skipped_rather_than_read_as_zero(history):
    blank = fc.TermActuals("Fall 2023", 0.0, 0, 0.0)
    result = fc.forecast([blank, *history], "Fall 2026", lookback=8)
    assert "Fall 2023" not in result.basis


def test_a_term_with_no_dues_forecasts_zero_dues():
    no_dues = fc.TermActuals("Fall 2025", 0.0, 0, 500.0, {MEETING_FOOD: 300.0})
    result = fc.forecast([no_dues], "Fall 2026")
    assert result.line("Dues income").expected == 0
    assert result.dues_rate == 0
    assert result.line("Net").expected == pytest.approx(200.0)


# --- Backtest ------------------------------------------------------------------


def test_backtest_only_looks_backward(history):
    tested = fc.backtest(history)
    assert "Fall 2024" not in set(tested.summary["Semester"])  # nothing before it
    for row in tested.summary.itertuples(index=False):
        order = [t.semester for t in history]
        used = row[1].split(", ")
        assert all(order.index(s) < order.index(row.Semester) for s in used)


def test_backtest_of_a_steady_history_is_exact():
    steady = [
        fc.TermActuals(f"Fall {year}", 400.0, 10, 100.0, {MEETING_FOOD: 300.0}, 50.0)
        for year in (2022, 2023, 2024, 2025)
    ]
    tested = fc.backtest(steady)
    assert tested.mean_absolute_error == pytest.approx(0.0)
    assert tested.share_in_range == pytest.approx(100.0)


def test_backtest_counts_a_committee_the_forecast_missed():
    terms = [
        fc.TermActuals("Fall 2024", 0.0, 0, 0.0, {MEETING_FOOD: 100.0}),
        fc.TermActuals("Fall 2025", 0.0, 0, 0.0, {MEETING_FOOD: 100.0, MERCH: 900.0}),
    ]
    detail = fc.backtest(terms).detail
    merch = detail[detail["Line"] == "Merch"].iloc[0]
    assert merch["Expected"] == 0 and merch["Actual"] == 900.0
    assert not merch["In range"]


# --- Utilization, targets, summary ----------------------------------------------


def test_utilization_is_percent_of_budget_and_blank_without_one(history):
    table = fc.budget_utilization(history).set_index("Committee")
    assert table.loc["Meeting Food", "Fall 2024"] == pytest.approx(50.0)
    assert table.loc["Meeting Food", "Fall 2025"] == pytest.approx(140.0)
    assert pd.isna(table.loc["Meeting Food", "Spring 2025"])
    assert pd.isna(table.loc["Merch", "Fall 2025"])  # a zero budget is not 0%


def test_upcoming_terms_extend_past_the_defined_ones():
    assert fc.upcoming_semesters(TERMS, AS_OF) == ["Fall 2026", "Spring 2027", "Fall 2027"]
    assert fc.default_target(TERMS, AS_OF) == "Spring 2027"


def test_executive_summary_names_the_numbers(history):
    result = fc.forecast(history, "Spring 2027")
    text = fc.executive_summary(result, fc.backtest(history))
    assert "Spring 2027" in text
    assert "Meeting Food" in text
    assert "Spring 2025, Spring 2026" in text
