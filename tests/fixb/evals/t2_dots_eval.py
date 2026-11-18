def _ev_box_sides_count(state):
    """Return list of (box, number_of_undrawn_sides) for boxes not yet owned, using board data."""
    lines, line_index, box_sides, line_boxes = _board(state)
    drawn = state["drawn"]
    res = []
    for box, sides in box_sides.items():
        if state["owner"][box] is None:
            n = 0
            for s in sides:
                if not drawn[s]:
                    n += 1
            res.append((box, n))
    return res, box_sides, line_boxes


def _ev_chain_parity(state, box_sides, line_boxes, boxes_info):
    """Rough estimate of long-chain parity advantage: count undrawn-side (3-sided->capturable) structure."""
    drawn = state["drawn"]
    undrawn_boxes = [b for b, n in boxes_info if n > 0]
    # Count capturable boxes (exactly one undrawn side -> immediate capture for mover)
    capturable = sum(1 for b, n in boxes_info if n == 1)
    # Undrawn lines that are safe (do not create a 3-sided box): approximate safe-move count
    safe = 0
    for i, d in enumerate(drawn):
        if d:
            continue
        ok = True
        for box in line_boxes[i]:
            if state["owner"][box] is not None:
                continue
            cnt = 0
            for s in box_sides[box]:
                if not drawn[s]:
                    cnt += 1
            if cnt == 2:  # drawing this would leave the box with 1 side -> gives a box away
                ok = False
                break
        if ok:
            safe += 1
    return capturable, safe


def evaluate(state, seat):
    opp = 1 - seat
    total = state["cols"] * state["rows"]
    my = state["score"][seat]
    op = state["score"][opp]
    if is_over(state):
        if my > op:
            return 1.0
        if op > my:
            return -1.0
        return 0.0

    boxes_info, box_sides, line_boxes = _ev_box_sides_count(state)
    capturable, safe = _ev_chain_parity(state, box_sides, line_boxes, boxes_info)

    remaining = total - my - op
    # Material lead, normalised by boxes still in play (scaled so it stays inside [-1,1]).
    material = (my - op) / float(total)

    # Whose move is it: the side to move can take capturable boxes; if no safe moves remain,
    # the side to move is in zugzwang and loses control.
    mover = state["turn"] % 2
    tempo = 0.0
    if safe == 0 and remaining > 0:
        tempo = -0.15 if mover == seat else 0.15
    elif capturable > 0:
        tempo = 0.1 if mover == seat else -0.1

    # Parity of safe moves: more safe moves with the seat to move favours the opponent of mover
    # in the endgame; keep the heuristic light.
    parity = 0.0
    if remaining > 0 and safe > 0:
        parity = (0.05 if (safe % 2 == 1) == (mover == seat) else -0.05)

    score = material + tempo + parity
    if score > 1.0:
        score = 1.0
    elif score < -1.0:
        score = -1.0
    return float(score)
