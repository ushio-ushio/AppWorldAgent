from agentlab.evaluation.report import parse_report_counts


def test_parse_appworld_evaluator_counts():
    assert parse_report_counts("Num Passed Tests : 5\nNum Total  Tests : 6") == (5, 6)


def test_incomplete_evaluator_report_keeps_missing_counts_unknown():
    assert parse_report_counts("Num Passed Tests : 2") == (2, None)
