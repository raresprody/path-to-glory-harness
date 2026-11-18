def _ev_val(p):
    return {"P": 1.0, "N": 3.0, "R": 5.0, "Q": 9.0, "K": 0.0}.get(p.upper(), 0.0)


def _ev_pst_bonus(p, f, r, n):
    # pawns: reward advancement; knights/king: reward centralization
    c = (n - 1) / 2.0
    cen = 1.0 - (abs(f - c) + abs(r - (n + 1) / 2.0)) / (n)
    u = p.upper()
    if u == "P":
        rank_from_home = (r - 2) if p.isupper() else (n - 1 - r)
        return 0.05 * max(0, rank_from_home)
    if u == "N":
        return 0.1 * cen
    if u == "Q":
        return 0.03 * cen
    return 0.0


def evaluate(state, seat):
    n = state["config"]["files"]
    board = state["board"]
    over = state["over"]
    if over:
        w = state["winner"]
        if w is None:
            return 0.0
        return 1.0 if w == seat else -1.0

    mat = 0.0
    my_pieces = 0
    opp_pieces = 0
    my_king = None
    opp_king = None
    my_pawn_files = set()
    opp_pawn_files = set()
    my_pawns = []
    opp_pawns = []
    for key, p in board.items():
        f_s, r_s = key.split(",")
        f, r = int(f_s), int(r_s)
        mine = (p.isupper() and seat == 0) or (p.islower() and seat == 1)
        val = _ev_val(p) + _ev_pst_bonus(p, f, r, n)
        u = p.upper()
        if u == "K":
            if mine:
                my_king = (f, r)
            else:
                opp_king = (f, r)
            continue
        if mine:
            mat += val
            my_pieces += 1
            if u == "P":
                my_pawn_files.add(f)
                my_pawns.append((f, r))
        else:
            mat -= val
            opp_pieces += 1
            if u == "P":
                opp_pawn_files.add(f)
                opp_pawns.append((f, r))

    score = mat / 40.0

    # passed pawns and doubled pawns (cheap approximations)
    for f, r in my_pawns:
        if f not in opp_pawn_files:
            score += 0.02
    for f, r in opp_pawns:
        if f not in my_pawn_files:
            score -= 0.02
    for f in my_pawn_files:
        cnt = sum(1 for (ff, _) in my_pawns if ff == f)
        if cnt > 1:
            score -= 0.03 * (cnt - 1)
    for f in opp_pawn_files:
        cnt = sum(1 for (ff, _) in opp_pawns if ff == f)
        if cnt > 1:
            score += 0.03 * (cnt - 1)

    # king safety: penalize king off back rank in middlegame, and nearness of enemy pieces
    if my_king is not None:
        home = 1 if seat == 0 else n
        if my_king[1] != home:
            score -= 0.03
    if opp_king is not None:
        home = 1 if seat == 1 else n
        if opp_king[1] != home:
            score += 0.03

    # endgame: fewer pieces with material edge is easier to convert; reward a king
    # approaching the enemy when little material remains
    if my_pieces + opp_pieces <= 4 and my_king is not None and opp_king is not None:
        dist = abs(my_king[0] - opp_king[0]) + abs(my_king[1] - opp_king[1])
        score += 0.02 * (2 * n - dist) / (2 * n)
        score -= 0.02 * 0  # placeholder for symmetry clarity

    # side to move: small tempo bonus
    if state["turn"] % 2 == seat:
        score += 0.01
    else:
        score -= 0.01

    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return float(score)
