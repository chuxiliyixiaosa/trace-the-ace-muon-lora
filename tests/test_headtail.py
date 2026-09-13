from trace_ace.headtail import HEADER, transform_content


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_short_prompt_is_unchanged():
    prompt = "objective" + HEADER + "student: 2\n"
    output, changed = transform_content(CharacterTokenizer(), prompt, max_tokens=100)
    assert output == prompt
    assert not changed


def test_long_prompt_keeps_both_ends_and_budget():
    prompt = "objective" + HEADER + "".join(f"turn {index:02d}: abcdefghij\n" for index in range(20))
    output, changed = transform_content(CharacterTokenizer(), prompt, max_tokens=220)
    assert changed
    assert "turn 00" in output
    assert "turn 19" in output
    assert "Middle dialogue turns were omitted" in output
    assert len(output) <= 220
