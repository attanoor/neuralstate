import uuid

from neuralstate.core.recall import Item, estimate_tokens, pack, render


def _item(section, text, score):
    return Item(section, uuid.uuid4(), text, score)


def test_rules_are_packed_before_facts_and_episodes():
    items = [_item("episodes", "e" * 40, 0.99), _item("facts", "f" * 40, 0.9), _item("rules", "r" * 40, 0.1)]
    chosen, _ = pack(items, budget_tokens=10_000)
    assert [i.section for i in chosen] == ["rules", "facts", "episodes"]


def test_budget_is_respected_and_long_items_skipped():
    items = [_item("rules", "short rule", 0.9), _item("facts", "x" * 4000, 0.9), _item("facts", "small fact", 0.5)]
    chosen, used = pack(items, budget_tokens=60)
    assert used <= 60
    assert [i.text for i in chosen] == ["short rule", "small fact"]


def test_best_score_first_within_section():
    items = [_item("facts", "low", 0.1), _item("facts", "high", 0.9)]
    chosen, _ = pack(items, 1000)
    assert [i.text for i in chosen] == ["high", "low"]


def test_render_groups_by_section_with_headings():
    chosen, _ = pack([_item("facts", "acme uses postgres", 1), _item("rules", "When deploying: avoid X", 1)], 1000)
    text = render(chosen)
    assert text.index("Learned rules") < text.index("Known facts")
    assert "- acme uses postgres" in text


def test_estimate_tokens_is_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd" * 10) == 10
