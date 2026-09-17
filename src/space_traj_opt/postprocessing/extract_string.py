import re
from collections.abc import Sequence

_PARAMETER_RE = re.compile(r"<([^<>]+)>")


def extract_parameterized_string(
    names: str | Sequence[str],
) -> list[str]:
    """
    Expand parameterized strings.

    Examples
    --------
    pos_<0-2>   -> ['pos_0', 'pos_1', 'pos_2']
    pos[<0-1>]  -> ['pos[0]', 'pos[1]']
    pos[0]      -> ['pos[0]']
    pos_<x,y,z> -> ['pos_x', 'pos_y', 'pos_z']
    """
    if isinstance(names, str):
        names = [names]

    result = []

    for name in names:
        match = _PARAMETER_RE.search(name)

        if match is None:
            result.append(name)
            continue

        expression = match.group(1)

        if "," in expression:
            values = [x.strip() for x in expression.split(",")]

        elif re.fullmatch(r"-?\d+\s*-\s*-?\d+", expression):
            start, end = map(int, re.findall(r"-?\d+", expression))
            step = 1 if end >= start else -1
            values = map(str, range(start, end + step, step))

        else:
            values = [expression]

        prefix = name[:match.start()]
        suffix = name[match.end():]

        result.extend(
            f"{prefix}{value}{suffix}"
            for value in values
        )

    return result