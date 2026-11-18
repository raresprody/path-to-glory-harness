def _ev_lines(board, rows, cols, win):
    """Yield all windows of length `win` as lists of (r, c) cells."""
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    for r in range(rows):
        for c in range(cols):
            for dr, dc in dirs:
                er, ec = r + dr * (win - 1), c + dc * (win - 1)
                if 0 <= er < rows and 0 <= ec < cols:
                    yield [(r + dr * i, c + dc * i) for i in range(win)]


def _ev_playable_rows(board, rows, cols):
    """For each column, the row index where the next disc would land (or None if full)."""
    heights = []
    for c in range(cols):
        h = None
        for r in range(rows):
            if board[r][c] is None:
                h = r
                break
        heights.append(h)
    return heights


def evaluate(state, seat):
    rows, cols, win = state['rows'], state['cols'], state['win']
    board = state['board']
    opp = 1 - seat

    # Terminal positions
    if state['over']:
        w = state['winner']
        if w == seat:
            return 1.0
        if w == opp:
            return -1.0
        return 0.0

    my_score = 0.0
    opp_score = 0.0
    my_threat_rows = set()
    opp_threat_rows = set()
    heights = _ev_playable_rows(board, rows, cols)

    for cells in _ev_lines(board, rows, cols, win):
        mine = opp_cnt = 0
        theirs = 0
        empties = []
        for (r, c) in cells:
            v = board[r][c]
            if v is None:
                empties.append((r, c))
            elif v == seat:
                mine += 1
            else:
                theirs += 1
        if mine and theirs:
            continue
        if mine == win:
            return 1.0
        if theirs == win:
            return -1.0
        if theirs == 0 and mine > 0:
            weight = (0, 1, 10, 60)[mine] if mine < 4 else 0
            my_score += weight
            if mine == win - 1 and len(empties) == 1:
                r, c = empties[0]
                my_threat_rows.add((r, c))
        elif mine == 0 and theirs > 0:
            weight = (0, 1, 10, 60)[theirs] if theirs < 4 else 0
            opp_score += weight
            if theirs == win - 1 and len(empties) == 1:
                r, c = empties[0]
                opp_threat_rows.add((r, c))

    # Immediate threats that are playable now are decisive-ish
    my_playable = 0
    opp_playable = 0
    for (r, c) in my_threat_rows:
        if heights[c] == r:
            my_playable += 1
    for (r, c) in opp_threat_rows:
        if heights[c] == r:
            opp_playable += 1
    if opp_playable >= 2:
        return -0.95
    if my_playable >= 1 and opp_playable == 0:
        return 0.8 if my_playable == 1 else 0.95
    if opp_playable == 1:
        return -0.6 if my_playable == 0 else 0.0

    # Center column preference
    center = cols // 2
    center_bonus = 0.0
    for r in range(rows):
        v = board[r][center]
        if v == seat:
            center_bonus += 3
        elif v == opp:
            center_bonus -= 3
    total = my_score - opp_score + center_bonus
    scale = 300.0
    val = total / (total + scale) if total >= 0 else -(-total) / (-total + scale)
    return max(-1.0, min(1.0, val * 0.9))
