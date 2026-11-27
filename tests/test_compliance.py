import numpy as np
import pytest

from conftest import make_det
from siteguard.compliance import ComplianceConfig, containment, evaluate, iou_matrix, relative_height


def test_containment_and_relative_height():
    items = np.array([[10, 10, 20, 20], [0, 0, 10, 10]], float)
    people = np.array([[0, 0, 100, 100], [15, 0, 100, 100]], float)
    c = containment(items, people)
    assert c[0, 0] == pytest.approx(1.0) and c[0, 1] == pytest.approx(0.5) and c[1, 1] == 0
    assert relative_height(items, people)[0, 0] == pytest.approx(0.15)
    assert containment(np.zeros((0, 4)), people).shape == (0, 2)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[5, 0, 15, 10]], float))[0, 0] == pytest.approx(1 / 3)


def test_compliant_worker():
    rep = evaluate(make_det(("Person", 0, 0, 100, 300), ("helmet", 20, -5, 80, 40), ("vest", 10, 80, 90, 180)))
    p = rep.people[0]
    assert p.status == "compliant"
    assert p.items["helmet"].status == "ok" and p.items["vest"].status == "ok"
    assert not rep.has_violation


def test_explicit_missing_helmet_is_violation():
    rep = evaluate(make_det(("Person", 0, 0, 100, 300), ("no_helmet", 25, 5, 75, 50), ("vest", 10, 80, 90, 180)))
    assert rep.people[0].status == "violation"
    assert rep.people[0].items["helmet"].status == "missing"
    assert rep.counts == {"compliant": 0, "violation": 1, "unverified": 0}


def test_not_visible_is_unverified_unless_strict():
    det = make_det(("Person", 0, 0, 100, 300), ("vest", 10, 80, 90, 180))  # head not detected at all
    assert evaluate(det).people[0].status == "unverified"
    assert evaluate(det, ComplianceConfig(strict=True)).people[0].status == "violation"


def test_helmet_held_in_hand_does_not_count():
    # helmet box at waist height: inside the person box, but not on the head
    rep = evaluate(make_det(("Person", 0, 0, 100, 300), ("helmet", 70, 140, 110, 175), ("vest", 10, 80, 90, 180)))
    assert rep.people[0].items["helmet"].status == "unknown"


def test_conflicting_evidence_uses_higher_confidence():
    det = make_det(
        ("Person", 0, 0, 100, 300),
        ("helmet", 20, 0, 80, 40, 0.4),
        ("no_helmet", 20, 0, 80, 45, 0.8),
        ("vest", 10, 80, 90, 180),
    )
    assert evaluate(det).people[0].items["helmet"].status == "missing"


def test_each_item_goes_to_one_person_in_a_crowd():
    # Front worker's helmet lies fully inside the box of a worker standing behind,
    # at that worker's chest height. Containment alone would give it to the worker
    # behind (100% vs 83%); the body-region test gives it to the right person.
    det = make_det(
        ("Person", 100, 150, 300, 750),  # front worker
        ("Person", 40, 20, 260, 380),  # worker behind, overlapping
        ("helmet", 150, 140, 220, 200),  # on the front worker's head
    )
    rep = evaluate(det, ComplianceConfig(required=("helmet",)))
    assert rep.people[0].items["helmet"].status == "ok"
    assert rep.people[1].items["helmet"].status == "unknown"


def test_unassigned_violation_is_reported():
    rep = evaluate(make_det(("no_helmet", 500, 500, 540, 540)))
    assert rep.people == [] and rep.has_violation
    assert rep.unassigned_violations[0]["class"] == "no_helmet"


def test_low_confidence_detections_are_ignored():
    det = make_det(("Person", 0, 0, 100, 300, 0.9), ("no_helmet", 25, 5, 75, 50, 0.1))
    assert evaluate(det, ComplianceConfig(min_confidence=0.25)).people[0].items["helmet"].status == "unknown"


def test_custom_required_items_and_validation():
    det = make_det(("Person", 0, 0, 100, 300), ("helmet", 20, 0, 80, 40), ("no_goggle", 35, 20, 65, 35))
    rep = evaluate(det, ComplianceConfig(required=("helmet", "goggles")))
    assert rep.people[0].status == "violation"
    with pytest.raises(ValueError):
        ComplianceConfig(required=("helmet", "cape"))


def test_report_serialisation():
    d = evaluate(make_det(("Person", 0, 0, 100, 300), ("helmet", 20, 0, 80, 40))).to_dict()
    assert d["summary"]["people"] == 1
    assert d["people"][0]["items"]["helmet"]["status"] == "ok"


def test_per_class_confidence_override():
    det = make_det(("Person", 0, 0, 100, 300, 0.9), ("no_helmet", 25, 5, 75, 50, 0.15))
    assert evaluate(det).people[0].items["helmet"].status == "unknown"
    cfg = ComplianceConfig(class_confidence={"no_helmet": 0.1})
    assert evaluate(det, cfg).people[0].items["helmet"].status == "missing"


def test_config_from_file(tmp_path):
    import json

    p = tmp_path / "t.json"
    p.write_text(json.dumps({"min_confidence": 0.3, "class_confidence": {"no_helmet": 0.1}}))
    cfg = ComplianceConfig.from_file(p, strict=True)
    assert cfg.min_confidence == 0.3 and cfg.strict and cfg.detector_confidence == 0.1
    assert ComplianceConfig.from_file(tmp_path / "missing.json").class_confidence == {}
