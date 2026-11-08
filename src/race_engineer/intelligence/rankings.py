"""Current ranking relationships and the telemetry needed to recheck them."""

from race_engineer.core.intelligence import ContextPacket


def ranking_checks(context: ContextPacket) -> dict[str, bool]:
    if context.temporal_scope not in {None, "current"}:
        return {}
    player = next(
        (
            item.value
            for item in context.evidence
            if item.kind != "unknown" and item.subject == "player" and item.metric == "position"
        ),
        None,
    )
    if isinstance(player, bool) or not isinstance(player, (int, float)) or player < 1:
        return {}
    checks = {"first": player == 1}
    last = next(
        (
            item.value
            for item in context.evidence
            if item.kind != "unknown" and item.subject == "player" and item.metric == "is_last"
        ),
        None,
    )
    if isinstance(last, bool):
        checks["last"] = last
    else:
        field = next(
            (
                item.value
                for item in context.evidence
                if item.kind != "unknown"
                and item.subject == "field"
                and item.metric == "position"
                and item.unit != "cars"
            ),
            None,
        )
        if not isinstance(field, bool) and isinstance(field, (int, float)) and field >= player:
            checks["last"] = player == field
    return checks
