import copy
import json
import re

from dsp import brief, demo


def test_structured_trees_stay_inside_one_factor(demo_out):
    assert demo_out["checks"]["structured_trees_mixing_factors"] == 0


def test_unconstrained_trees_do_mix_factors(demo_out):
    # Shows the one-factor check can fail: without constraints, trees mix factors.
    assert demo_out["checks"]["unconstrained_trees_mixing_factors"] > 0


def test_sanity_factor_shares_add_up_to_the_score(demo_out):
    # True for any tree model (TreeSHAP); a sanity check, not the benefit.
    assert demo_out["checks"]["sanity_max_abs_gap_factor_sum_vs_score"] < 1e-6
    assert demo_out["checks"]["sanity_all_records_add_up"]


def test_structured_model_is_separable_by_factor(demo_out):
    # Changing one factor's facts never moves another factor's share.
    for factor, r in demo_out["checks"]["separation_structured"].items():
        assert r["share_with_other_factors_moved"] == 0, factor
        assert r["max_other_factor_move"] <= 1e-9, factor


def test_unconstrained_model_is_not_separable(demo_out):
    # Shows the separation test can fail: interactions cross factors.
    moved = [r["share_with_other_factors_moved"]
             for r in demo_out["checks"]["separation_unconstrained"].values()
             if r["max_own_factor_move"] > 0]
    assert max(moved) > 0.5


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
