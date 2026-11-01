"""Advisor brief from a decision record, by fixed templates.

Nothing here computes or invents a reason: every fact printed is a field of
the record, and labels come from the reasoning model file.
"""


UNINFORMATIVE = (None, "unknown", "nonexistent")


def _value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _facts(factor, reasoning, informative_only=False) -> str:
    parts = []
    for name, value in factor["features"].items():
        if informative_only and value in UNINFORMATIVE:
            continue
        label = reasoning.feature_labels.get(name, name)
        parts.append(f"{label} {_value(value)}")
    return ", ".join(parts)


def _label(factor, reasoning) -> str:
    return reasoning.factors[factor["id"]]["label"]


def render(record, reasoning) -> str:
    s = record["score"]
    lines = [f"### Client {record['client_ref']}", ""]

    lines.append(
        f"**Model estimate:** about {s['probability_pct']}% of clients like this say yes "
        f"(calibrated on {record['model']['calibrated_on']}). "
        f"Ranks in the top {s['rank_top_pct']}% of this call list, "
        f"{'inside' if s['in_top'] else 'outside'} the top-{s['threshold_top_pct']}% cut-off."
    )
    if record["abstain"]:
        lines.append("**Unsure:** the model's range crosses the call threshold. Check first.")
    lines.append("")

    reasons = sorted((f for f in record["factors"] if not f["context_only"]),
                     key=lambda f: f["contribution"], reverse=True)
    up = [f for f in reasons if f["contribution"] > 0][:2]
    down = [f for f in reversed(reasons) if f["contribution"] < 0][:1]
    def reason(f):
        text = f"{_label(f, reasoning).lower()} ({_facts(f, reasoning)})"
        if reasoning.factors[f["id"]].get("sensitive"):
            text += " [model reason only, not to raise with the client]"
        return text

    if up:
        lines.append("**What raises the score most** (against the average client in the "
                     f"training months): {'; '.join(reason(f) for f in up)}.")
    if down:
        lines.append(f"**What holds it down:** {reason(down[0])}.")

    context = [f for f in record["factors"] if f["context_only"]]
    for f in context:
        if f["contribution"] != 0:
            direction = "raised" if f["contribution"] > 0 else "lowered"
            lines.append(
                f"**Context:** market conditions at the time {direction} the score "
                f"({_facts(f, reasoning)}). This is context, not a reason the client wants the product."
            )

    lines.append("")
    talk = [f for f in record["factors"]
            if f["talking_point"] and f["contribution"] > 0
            and _facts(f, reasoning, informative_only=True)]
    if talk:
        lines.append("**What you could raise in the call:**")
        for f in talk:
            lines.append(f"- {_label(f, reasoning)}: {_facts(f, reasoning, informative_only=True)}")
    else:
        lines.append("**What you could raise in the call:** no relationship or capacity facts stand out.")

    lines.append("")
    lines.append(
        "_The reasons describe the model, not the client's motives. Factor shares add up "
        "exactly to the model's score for this client (check passed: "
        f"{'yes' if record['check']['exact'] else 'NO'})._"
    )
    return "\n".join(lines)
