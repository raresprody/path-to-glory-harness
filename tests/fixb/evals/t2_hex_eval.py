def _ev_path_cost(board, n, player):
    # 0-1 BFS: minimum number of empty cells needed to connect player's edges.
    # Own stones cost 0, empty cells cost 1, opponent stones are blocked.
    from collections import deque
    INF = 10 ** 9
    dist = [[INF] * n for _ in range(n)]
    dq = deque()
    if player == 0:
        starts = [(0, c) for c in range(n)]
        done = lambda r, c: r == n - 1
    else:
        starts = [(r, 0) for r in range(n)]
        done = lambda r, c: c == n - 1
    for r, c in starts:
        v = board[r][c]
        if v == 1 - player:
            continue
        w = 0 if v == player else 1
        if w < dist[r][c]:
            dist[r][c] = w
            if w == 0:
                dq.appendleft((r, c))
            else:
                dq.append((r, c))
    best = INF
    while dq:
        r, c = dq.popleft()
        d = dist[r][c]
        if done(r, c):
            if d < best:
                best = d
            continue
        for nr, nc in _ev_nbrs(r, c, n):
            v = board[nr][nc]
            if v == 1 - player:
                continue
            w = d + (0 if v == player else 1)
            if w < dist[nr][nc]:
                dist[nr][nc] = w
                if w == d:
                    dq.appendleft((nr, nc))
                else:
                    dq.append((nr, nc))
    return best


def _ev_nbrs(r, c, n):
    for dr, dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc


def _ev_connected(board, n, player):
    if player == 0:
        starts = [(0, c) for c in range(n) if board[0][c] == 0]
        done = lambda r, c: r == n - 1
    else:
        starts = [(r, 0) for r in range(n) if board[r][0] == 1]
        done = lambda r, c: c == n - 1
    seen = set(starts)
    stack = list(starts)
    while stack:
        r, c = stack.pop()
        if done(r, c):
            return True
        for nr, nc in _ev_nbrs(r, c, n):
            if (nr, nc) not in seen and board[nr][nc] == player:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return False


def evaluate(state, seat):
    n = state['n']
    board = state['board']
    opp = 1 - seat

    if state['over']:
        w = state['winner']
        if w == seat:
            return 1.0
        if w == opp:
            return -1.0
        return 0.0

    my_cost = _ev_path_cost(board, n, seat)
    op_cost = _ev_path_cost(board, n, opp)
    INF = 10 ** 9
    if my_cost >= INF and op_cost >= INF:
        return 0.0

    # Shortest-path "resistance": fewer empty cells needed is better.
    my_c = my_cost if my_cost < INF else 2 * n
    op_c = op_cost if op_cost < INF else 2 * n
    diff = (op_c - my_c) / float(n)

    # Stone count / tempo: small bonus for stones placed in own useful region.
    my_stones = 0
    op_stones = 0
    center = (n - 1) / 2.0
    center_bonus = 0.0
    for r in range(n):
        row = board[r]
        for c in range(n):
            v = row[c]
            if v is None:
                continue
            d = abs(r - center) + abs(c - center)
            weight = (n - d) / float(n)
            if v == seat:
                my_stones += 1
                center_bonus += weight
            else:
                op_stones += 1
                center_bonus -= weight
    center_term = center_bonus / float(n * n)

    score = 0.8 * diff + 0.2 * center_term
    if score > 1.0:
        score = 1.0
    elif score < -1.0:
        score = -1.0
    return float(score)
