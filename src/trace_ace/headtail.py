from __future__ import annotations


HEADER = "\n\nFull dialogue:\n"
OMISSION_NOTE = (
    "[context note] Middle dialogue turns were omitted to fit the context window; "
    "timestamps and retained turn text are unchanged.\n"
)


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _choose_prefix(lengths: list[int], budget: int) -> int:
    used = 0
    count = 0
    for length in lengths:
        if used + length > budget:
            break
        used += length
        count += 1
    return count


def _choose_suffix(lengths: list[int], start: int, budget: int) -> int:
    used = 0
    index = len(lengths)
    while index > start and used + lengths[index - 1] <= budget:
        used += lengths[index - 1]
        index -= 1
    return index


def transform_content(
    tokenizer,
    content: str,
    max_tokens: int = 10_000,
    head_fraction: float = 0.35,
) -> tuple[str, bool]:
    """Keep the objective prefix and allocate long-dialogue tokens 35%/65%."""
    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must lie strictly between zero and one")
    if token_count(tokenizer, content) <= max_tokens:
        return content, False

    before, marker, dialogue = content.partition(HEADER)
    if not marker:
        raise ValueError("prompt is missing the full-dialogue header")
    prefix = before + HEADER
    lines = dialogue.splitlines(keepends=True)
    if not lines:
        raise ValueError("prompt contains no dialogue turns")
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    budget = max_tokens - token_count(tokenizer, prefix) - token_count(tokenizer, OMISSION_NOTE)
    if budget <= 0:
        raise ValueError("prompt prefix exceeds the content-token budget")

    lengths = [token_count(tokenizer, line) for line in lines]
    head_count = _choose_prefix(lengths, int(budget * head_fraction))
    suffix_start = _choose_suffix(lengths, head_count, budget - sum(lengths[:head_count]))

    def assemble() -> str:
        return prefix + "".join(lines[:head_count]) + OMISSION_NOTE + "".join(lines[suffix_start:])

    transformed = assemble()
    while token_count(tokenizer, transformed) > max_tokens:
        if head_count:
            head_count -= 1
        elif suffix_start < len(lines):
            suffix_start += 1
        else:
            raise ValueError("could not satisfy the content-token budget")
        transformed = assemble()
    if head_count >= suffix_start:
        raise ValueError("head and tail selections overlap")
    return transformed.rstrip("\n"), True
