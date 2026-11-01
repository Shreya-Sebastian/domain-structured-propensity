import copy
import json
import re

from dsp import brief, demo


def test_structured_trees_stay_inside_one_factor(demo_out):
    assert demo_out["checks"]["structured_trees_mixing_factors"] == 0


def test_unconstrained_trees_do_mix_factors(demo_out):
    # Shows the one-factor check can fail: without constraints, trees mix factors.
    assert demo_out["checks"]["unconstrained_trees_mixing_factors"] > 0


def test_factor_contributions_sum_exactly_to_the_score(demo_out):
    assert demo_out["checks"]["max_abs_gap_factor_sum_vs_score"] < 1e-6
    assert demo_out["checks"]["all_records_exact"]


def _numbers(text):
    return {float(t) for t in re.findall(r"\d+(?:\.\d+)?", text)}


def test_every_number_in_a_brief_comes_from_its_record(demo_out):
    rm = demo_out["reasoning"]
    for record in demo.example_records(demo_out["records"]):
        text = brief.render(record, rm)
        assert _numbers(text) <= _numbers(json.dumps(record)), record["client_ref"]


def test_life_stage_is_never_a_talking_point(demo_out):
    rm = demo_out["reasoning"]
    record = copy.deepcopy(demo_out["records"][0])
    for f in record["factors"]:
        if f["id"] == "life_stage":
            f["contribution"] = 5.0  # make it the strongest reason
    text = brief.render(record, rm)
    talk = text.split("What you could raise in the call")[1]
    assert "Life stage" not in talk
    assert "model reason only" in text


def test_brief_only_uses_record_fields(demo_out):
    rm = demo_out["reasoning"]
    record = copy.deepcopy(demo_out["records"][0])
    del record["score"]
    try:
        brief.render(record, rm)
    except KeyError:
        return
    raise AssertionError("brief rendered without the score field")
