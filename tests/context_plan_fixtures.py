"""Build v5 wire fixtures from the terse selections used by adapter/execution tests.

This is test data construction, not a legacy-format path in the production adapter.
Explicit v5 drafts (including malformed drafts) pass through unchanged.
"""


def wire_plan(raw: object) -> object:
    if not isinstance(raw, dict) or "social_comment" in raw:
        return raw
    draft = dict(raw)
    purpose = draft.pop("purpose", "race_information")
    draft["social_comment"] = purpose != "race_information"
    draft["requested_facts"] = [] if purpose == "social" else ["authored test request"]
    return draft
