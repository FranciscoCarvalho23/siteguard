from collections import Counter

import numpy as np

from conftest import make_det
from siteguard.compliance import ComplianceConfig
from siteguard.metrics import compliance_confusion, match_people, summarize


def test_match_people_greedy():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    pred = np.array([[21, 21, 31, 31], [0, 0, 10, 9], [100, 100, 110, 110]], float)
    assert match_people(gt, pred) == {0: 1, 1: 0}
    assert match_people(np.zeros((0, 4)), pred) == {}


def test_summarize():
    c = Counter({("missing", "missing"): 3, ("missing", "ok"): 1, ("ok", "missing"): 1, ("ok", "ok"): 5})
    s = summarize(c)
    assert s["violation_recall"] == 0.75 and s["violation_precision"] == 0.75
    assert s["false_alarm_rate"] == round(1 / 6, 3)
    assert 0 < s["violation_f2"] <= 1


def test_compliance_confusion_end_to_end():
    gt = make_det(
        ("Person", 0, 0, 100, 300),
        ("no_helmet", 25, 5, 75, 50),
        ("Person", 200, 0, 300, 300),
        ("helmet", 220, 0, 280, 40),
    )
    # prediction finds both workers but misses the no_helmet box
    pred = make_det(("Person", 2, 0, 100, 300), ("Person", 200, 2, 300, 300), ("helmet", 220, 0, 280, 40))
    conf, total, found = compliance_confusion([(gt, pred)], ComplianceConfig(required=("helmet",)))
    assert total == 2 and found == 2
    assert conf["helmet"] == Counter({("missing", "unknown"): 1, ("ok", "ok"): 1})


def test_silent_miss_and_strict_recall():
    c = Counter({("missing", "missing"): 6, ("missing", "unknown"): 2, ("missing", "ok"): 2})
    s = summarize(c)
    assert s["silent_miss_rate"] == 0.2 and s["strict_recall"] == 0.8


def test_spurious_worker_counts_as_false_alarm():
    gt = make_det(("Person", 0, 0, 100, 300), ("no_helmet", 25, 5, 75, 50))
    # correct flag on the real worker + a phantom "worker" (empty seat) flagged as well
    pred = make_det(
        ("Person", 0, 0, 100, 300),
        ("no_helmet", 25, 5, 75, 50),
        ("Person", 500, 0, 600, 300),
        ("no_helmet", 525, 5, 575, 50),
    )
    conf, _, _ = compliance_confusion([(gt, pred)], ComplianceConfig(required=("helmet",)))
    s = summarize(conf["helmet"])
    assert s["violation_recall"] == 1.0 and s["violation_precision"] == 0.5
    assert s["spurious_worker_flags"] == 1
