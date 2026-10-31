# Problem-Set-3
**Model:** Claude ~ Opus 5.5 ; **Date:** 10/31

***Prompt:*** 
[Prep] Act as an analytics engineer who reviews SQL before it reaches
a dashboard.

[Request] Write a Python function review_query(con, sql) that runs
against DuckDB and returns a list of warnings for a given SELECT. It
should flag at minimum: a join that changes the row count, a filter on
a nullable column that will silently drop unknowns, and an aggregate
computed over a fanned-out result.

[Iterate] duckdb and pandas only. Reuse my table names, customers and
orders. Return structured warnings, not printed text, so I can test it.

[Mechanism Check] Add comments explaining how you detect fan-out
without running the full join twice, and why three-valued logic makes
the null filter a correctness issue rather than a style issue.

[Evaluate] Run it against both of my Phase 1 queries and tell me
whether it would have caught the $1,240 before it reached the board.

***Reply:***
review_query.py is ready. Before you rely on it, though, I need to be upfront on two points. DuckDB couldn’t be installed in my sandbox because it has no network access, so I’ve only checked that the file compiles. I haven’t run it. Also, your two Phase 1 queries aren’t in this chat or this project. That means I can’t yet tell you whether it would have caught the $1,240.

How it works. review_query(con, sql) returns a list of QueryWarning objects with these fields: check, severity, message, where, and a details dict of counts. warnings_to_frame() turns that list into a DataFrame for tests.

The function never runs your query. It uses DuckDB’s own parser (json_serialize_sql) to read the query’s structure, then runs small counting queries built from pieces of it. It also checks CTEs and subqueries in FROM. If a CTE duplicates rows, the function remembers that, so a SUM in a later SELECT that reads the CTE still gets flagged. Bugs often hide there because the join and the aggregate are in different places.

The mechanisms (also explained in comments in the file):

Fan-out without running the join. A join produces L(k) × R(k) rows for each key k. So the function groups each side by its key once, joins the two small count tables, and adds up the products. That gives the exact output row count, how many rows go unmatched, and the most copies any single row gets. The aggregate check reuses those copy counts, so one counting query serves both checks.
Why the null filter matters. Any comparison with NULL, including <> and NOT IN, evaluates to UNKNOWN, and WHERE keeps only TRUE rows. So status <> 'refunded' also drops rows whose status is unknown. A WHERE on an orders column after customers LEFT JOIN orders turns that join into an INNER JOIN. These change the totals, not just how the query reads.

What it would catch, depending on where the $1,240 came from:

A customers-level amount summed after joining to orders. This gets flagged as fanout_aggregate (high severity), and the message shows how many times the rows were repeated.
Orders rows repeated because customer_id isn’t unique in customers. This gets flagged as join_row_count, with counts like “N rows in, M out,” plus fanout_aggregate on any SUM of order amounts.
Revenue lost to a <> filter on a column that has NULLs, or to a WHERE on the orders side of a LEFT JOIN. This gets flagged as null_filter, with the number of rows dropped.
Orders with a NULL or unmatched customer_id dropped by an INNER JOIN. This gets flagged as join_row_count with “drops rows.”
Not caught: a wrong date range, a wrong filter value, or a join on the wrong column that happens to be one-to-one. The SQL is valid in those cases, but the logic is wrong.

The __main__ block runs two toy queries on made-up data and lists the warnings each one should produce. If the output doesn’t match, a likely cause is that your DuckDB version names some parser fields differently. Paste me the traceback and I’ll adjust it.

Can you paste your two Phase 1 queries (and the rough size of customers and orders)? Then I can trace exactly which warning, if any, would have fired on the $1,240.


# CHANGED: removed `from __future__ import annotations` (must be the first line of a file; SyntaxError in a notebook cell)

import copy
import json
from dataclasses import asdict, dataclass, field

import duckdb
import pandas as pd


@dataclass
class QueryWarning:
    check: str                 # join_row_count | null_filter | fanout_aggregate | not_analyzed
    severity: str              # high | medium | info
    message: str
    where: str = "main query"  # "main query", "CTE <name>", "subquery <alias>"
    details: dict = field(default_factory=dict)


def warnings_to_frame(warnings: list[QueryWarning]) -> pd.DataFrame:
    cols = ["check", "severity", "where", "message", "details"]
    return pd.DataFrame([asdict(w) for w in warnings], columns=cols)


# Aggregates whose answer does not change when the same row appears twice.
_DUPLICATE_SAFE = {"min", "max", "any_value", "arbitrary", "first", "last",
                   "bool_and", "bool_or", "arg_min", "arg_max", "min_by", "max_by",
                   "bit_and", "bit_or"}

# Expressions that look at NULL on purpose, so a NULL inside them is not "silent".
_NULL_AWARE_TYPES = {"OPERATOR_IS_NULL", "OPERATOR_IS_NOT_NULL", "OPERATOR_COALESCE",
                     "COMPARE_DISTINCT_FROM", "COMPARE_NOT_DISTINCT_FROM"}
_NULL_AWARE_FUNCS = {"coalesce", "ifnull"}
_NEGATIONS = {"OPERATOR_NOT", "COMPARE_NOTEQUAL", "COMPARE_NOT_IN"}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def review_query(con: duckdb.DuckDBPyConnection, sql: str) -> list[QueryWarning]:
    node = _parse(con, sql)
    out: list[QueryWarning] = []
    _review_node(con, node, outer_ctes=[], where="main query", out=out, fanned={})
    return out


# --------------------------------------------------------------------------
# Parser helpers
# --------------------------------------------------------------------------
def _parse(con, sql):
    raw = con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
    tree = json.loads(raw)
    if tree.get("error"):
        raise ValueError(f"DuckDB could not parse this as a SELECT: {tree.get('error_message')}")
    if len(tree["statements"]) != 1:
        raise ValueError("review_query expects exactly one SELECT statement")
    return tree["statements"][0]["node"]


def _to_sql(con, node):
    payload = json.dumps({"error": False,
                          "statements": [{"node": node, "named_param_map": []}]})
    return con.execute("SELECT json_deserialize_sql(?::JSON)", [payload]).fetchone()[0]


def _select_from(con, root, ref, select_list=None):
    """SQL for `SELECT <select_list or *> FROM <ref>` that can still see root's CTEs."""
    node = _parse(con, "SELECT * FROM __review_placeholder__")  # skeleton for this DuckDB version
    node["from_table"] = copy.deepcopy(ref)
    node["cte_map"] = copy.deepcopy(root.get("cte_map") or {"map": []})
    if select_list is not None:
        node["select_list"] = copy.deepcopy(select_list)
    return _to_sql(con, node)


def _colref(con, alias, col):
    ref = _parse(con, "SELECT a.b FROM t")["select_list"][0]
    ref["column_names"] = [alias, col]
    return ref


def _named(expr, name):
    expr = copy.deepcopy(expr)
    expr["alias"] = name
    return expr


def _q(name):
    return '"' + name.replace('"', '""') + '"'


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _leaves(ref):
    if not ref:
        return []
    if ref.get("type") == "JOIN":
        return _leaves(ref["left"]) + _leaves(ref["right"])
    if ref.get("type") in ("BASE_TABLE", "SUBQUERY", "TABLE_FUNCTION"):
        return [ref]
    return []


def _joins(ref):
    if not ref or ref.get("type") != "JOIN":
        return []
    return _joins(ref["left"]) + _joins(ref["right"]) + [ref]


def _alias(ref):
    return (ref.get("alias") or ref.get("table_name") or "").lower()


def _and_terms(expr):
    if not expr:
        return []
    if expr.get("type") == "CONJUNCTION_AND":
        return [t for c in expr["children"] for t in _and_terms(c)]
    return [expr]


def _colkey(c):
    return tuple(n.lower() for n in c.get("column_names", []))


def _resolve(colref, scope):
    """Column reference -> (alias, column) in this SELECT, or None if unknown/ambiguous."""
    names = [n.lower() for n in colref.get("column_names", [])]
    if len(names) >= 2:
        alias, col = names[-2], names[-1]
        return (alias, col) if alias in scope and col in scope[alias]["columns"] else None
    if not names:
        return None
    hits = [a for a, s in scope.items() if names[0] in s["columns"]]
    return (hits[0], names[0]) if len(hits) == 1 else None


def _aggregates(expr, names):
    if isinstance(expr, list):
        for e in expr:
            yield from _aggregates(e, names)
        return
    if not isinstance(expr, dict):
        return
    if expr.get("class") == "FUNCTION" and expr.get("function_name", "").lower() in names:
        yield expr
        return
    if expr.get("class") in ("SUBQUERY", "WINDOW"):
        return
    for v in expr.values():
        if isinstance(v, (dict, list)):
            yield from _aggregates(v, names)


# --------------------------------------------------------------------------
# One SELECT (main query, a CTE, or a subquery in FROM)
# --------------------------------------------------------------------------
def _review_node(con, node, outer_ctes, where, out, fanned):
    """Review one SELECT. Returns info about columns it outputs duplicated, or None."""
    node = copy.deepcopy(node)
    if node.get("type") != "SELECT_NODE":
        out.append(QueryWarning("not_analyzed", "info",
                                f"{node.get('type')} (e.g. UNION) is not reviewed; "
                                "review each SELECT in it on its own.", where))
        return None

    # CTEs first, in order, so later SELECTs know which CTEs duplicate rows.
    node.setdefault("cte_map", {"map": []})
    own_ctes = node["cte_map"].get("map", [])
    for i, entry in enumerate(own_ctes):
        info = _review_node(con, entry["value"]["query"]["node"], outer_ctes + own_ctes[:i],
                            f"CTE {entry['key']}", out, fanned)
        if info:
            fanned[entry["key"].lower()] = info
    node["cte_map"]["map"] = outer_ctes + own_ctes
    cte_names = {e["key"].lower() for e in node["cte_map"]["map"]}

    # Every table / CTE / subquery this SELECT reads, with its columns.
    scope = {}
    for leaf in _leaves(node.get("from_table")):
        alias = _alias(leaf)
        upstream = None
        if leaf["type"] == "SUBQUERY":
            upstream = _review_node(con, leaf["subquery"]["node"], node["cte_map"]["map"],
                                    f"subquery {alias}", out, fanned)
        elif leaf["type"] == "BASE_TABLE":
            upstream = fanned.get(leaf.get("table_name", "").lower())
        try:
            is_table = leaf["type"] == "BASE_TABLE" and leaf.get("table_name", "").lower() not in cte_names
            if is_table:   # real table: DESCRIBE it directly so declared NOT NULL is visible
                name = ".".join(_q(p) for p in (leaf.get("catalog_name"), leaf.get("schema_name"),
                                                leaf["table_name"]) if p)
                desc = con.execute(f"DESCRIBE {name}").df()
            else:
                desc = con.execute(f"DESCRIBE SELECT * FROM ({_select_from(con, node, leaf)})").df()
        except Exception as e:  # noqa: BLE001 -- report, don't crash the review
            out.append(QueryWarning("not_analyzed", "info",
                                    f"Could not read the columns of {alias}: {e}", where))
            return None
        scope[alias] = {
            "ref": leaf,
            # column -> declared nullable (None when the source is a CTE/subquery)
            "columns": {c.lower(): ((n == "YES") if is_table else None)
                        for c, n in zip(desc["column_name"], desc["null"])},
            "upstream": upstream,  # columns already duplicated before this SELECT
        }

    duplicated = {}     # alias -> up to how many times each of its rows repeats here
    nullable_side = {}  # alias -> (join label, rows that join fills with NULL)
    for j in _joins(node.get("from_table")):
        _check_join(con, node, j, scope, where, out, duplicated, nullable_side)

    seen = set()
    for c, negated in _null_sensitive_columns(node.get("where_clause")):
        r = _resolve(c, scope)
        if r and r not in seen:
            seen.add(r)
            _check_null_filter(con, node, r, negated, scope, nullable_side, where, out)

    agg_names = {r[0] for r in con.execute(
        "SELECT DISTINCT lower(function_name) FROM duckdb_functions() "
        "WHERE function_type = 'aggregate'").fetchall()} | {"count_star"}
    upstream = {a: s["upstream"] for a, s in scope.items() if s["upstream"]}
    aggs = list(_aggregates(node.get("select_list", []) + [node.get("having")], agg_names))
    for agg in aggs:
        _check_aggregate(agg, scope, duplicated, upstream, where, out)

    # Does this SELECT hand duplicated rows to whoever reads it?
    if (not duplicated and not upstream) or node.get("group_expressions") or aggs:
        return None  # nothing duplicated, or rows are collapsed here (and reported above)
    cols, factor = set(), 1
    for item in node.get("select_list", []):
        if item.get("class") == "STAR":
            rel = (item.get("relation_name") or "").lower()
            for a, s in scope.items():
                if rel and a != rel:
                    continue
                if a in duplicated:
                    cols |= set(s["columns"]); factor = max(factor, duplicated[a])
                elif a in upstream:
                    cols |= upstream[a]["columns"]; factor = max(factor, upstream[a]["factor"])
            continue
        for c in (n for n in _walk(item) if n.get("class") == "COLUMN_REF"):
            r = _resolve(c, scope)
            if not r:
                continue
            a, col = r
            f = duplicated.get(a) or (upstream[a]["factor"]
                                      if a in upstream and col in upstream[a]["columns"] else None)
            out_name = (item.get("alias") or (col if item is c else "")).lower()
            if f and out_name:
                cols.add(out_name); factor = max(factor, f)
    return {"columns": cols, "factor": factor, "source": where} if cols else None


# --------------------------------------------------------------------------
# Check 1: joins that change the row count
# --------------------------------------------------------------------------
def _check_join(con, root, j, scope, where, out, duplicated, nullable_side):
    left = {_alias(x) for x in _leaves(j["left"])}
    right = {_alias(x) for x in _leaves(j["right"])}
    jtype = j.get("join_type", "INNER")
    label = f"{jtype} JOIN {'/'.join(sorted(left))} -> {'/'.join(sorted(right))}"
    if jtype in ("SEMI", "ANTI"):
        return  # these keep or drop left rows but never copy them

    pairs, extra_terms = [], 0
    for col in j.get("using_columns") or []:
        col = col.lower()
        la = [a for a in left if col in scope[a]["columns"]]
        ra = [a for a in right if col in scope[a]["columns"]]
        if len(la) == 1 and len(ra) == 1:
            pairs.append((_colref(con, la[0], col), _colref(con, ra[0], col)))
    for t in _and_terms(j.get("condition")):
        a, b = t.get("left"), t.get("right")
        if (t.get("type") == "COMPARE_EQUAL" and isinstance(a, dict) and isinstance(b, dict)
                and a.get("class") == b.get("class") == "COLUMN_REF"):
            ra, rb = _resolve(a, scope), _resolve(b, scope)
            if ra and rb and ra[0] in left and rb[0] in right:
                pairs.append((a, b)); continue
            if ra and rb and rb[0] in left and ra[0] in right:
                pairs.append((b, a)); continue
        extra_terms += 1

    if not pairs:
        for a in (right if jtype in ("LEFT", "OUTER") else set()) | (left if jtype in ("RIGHT", "OUTER") else set()):
            nullable_side[a] = (label, None)
        out.append(QueryWarning("not_analyzed", "info",
                                f"{label}: no `column = column` key found, so its row count was not checked.",
                                where))
        return

    s = _size_join(con, root, j, pairs)
    rows_in = s["right_rows"] if jtype == "RIGHT" else s["left_rows"]
    rows_out = (s["matched_rows"]
                + (s["left_unmatched"] if jtype in ("LEFT", "OUTER") else 0)
                + (s["right_unmatched"] if jtype in ("RIGHT", "OUTER") else 0))
    left_copies, right_copies = s["max_right_per_key"], s["max_left_per_key"]

    if jtype in ("LEFT", "OUTER"):
        for a in right:
            nullable_side[a] = (label, s["left_unmatched"])
    if jtype in ("RIGHT", "OUTER"):
        for a in left:
            nullable_side[a] = (label, s["right_unmatched"])
    if left_copies > 1:
        for a in left:
            duplicated[a] = duplicated.get(a, 1) * left_copies
    if right_copies > 1:
        for a in right:
            duplicated[a] = duplicated.get(a, 1) * right_copies

    if rows_out == rows_in and left_copies <= 1 and right_copies <= 1:
        return
    keys = ", ".join(".".join(l["column_names"]) + " = " + ".".join(r["column_names"]) for l, r in pairs)
    copied = []
    if left_copies > 1:
        copied.append(f"one {'/'.join(sorted(left))} row matches up to {left_copies} "
                      f"{'/'.join(sorted(right))} rows")
    if right_copies > 1:
        copied.append(f"one {'/'.join(sorted(right))} row matches up to {right_copies} "
                      f"{'/'.join(sorted(left))} rows")
    if rows_out > rows_in:
        sev, what = "high", f"fans out: {rows_in:,} rows in, {rows_out:,} rows out"
    elif rows_out < rows_in:
        sev, what = "medium", (f"drops rows: {rows_in:,} rows in, {rows_out:,} rows out "
                               "(keys with no match, NULL keys included)")
    else:
        sev, what = "medium", (f"row count is unchanged ({rows_in:,}) only because "
                               "duplicated rows and dropped rows cancel out")
    msg = f"{label} on {keys} {what}" + (f"; {'; '.join(copied)}" if copied else "") + "."
    if extra_terms:
        msg += (f" The ON clause has {extra_terms} extra condition(s) this count ignores, "
                "so treat the numbers as an upper bound.")
        sev = "medium" if sev == "high" else sev
    out.append(QueryWarning("join_row_count", sev, msg, where,
                            {**s, "rows_in": rows_in, "rows_out": rows_out,
                             "join_type": jtype, "keys": keys, "extra_on_terms": extra_terms}))


def _size_join(con, root, j, pairs):
    # HOW FAN-OUT IS DETECTED WITHOUT RUNNING THE FULL JOIN (TWICE OR AT ALL)
    #
    # A join's row count is decided entirely by how many rows share each key.
    # If key k appears L(k) times on the left and R(k) times on the right, the
    # join produces L(k) * R(k) rows for that key. So instead of building the
    # joined rows, each side is grouped by its key once (one pass per table,
    # one small row per distinct key), the two small count tables are joined,
    # and L(k) * R(k) is summed. Keys found on only one side give the rows a
    # LEFT / RIGHT / FULL join keeps (or an INNER join drops); NULL keys never
    # match, which is also what the real join does.
    #
    # The same pass also returns max R(k), "how many times can one left row be
    # copied", and max L(k) for the other direction. The fan-out aggregate check
    # reuses those numbers, so the join-row-count check and the aggregate check
    # share one counting query instead of each running the join.
    #
    # Limitation: counts use the tables before WHERE; WHERE can only remove rows.
    k = [f"k{i}" for i in range(len(pairs))]
    lsql = _select_from(con, root, j["left"], [_named(l, n) for (l, _), n in zip(pairs, k)])
    rsql = _select_from(con, root, j["right"], [_named(r, n) for (_, r), n in zip(pairs, k)])
    keys = ", ".join(k)
    on = " AND ".join(f"lc.{c} = rc.{c}" for c in k)
    q = f"""
        WITH _rv_l AS (SELECT * FROM ({lsql})),
             _rv_r AS (SELECT * FROM ({rsql})),
             lc AS (SELECT {keys}, COUNT(*) AS n FROM _rv_l GROUP BY {keys}),
             rc AS (SELECT {keys}, COUNT(*) AS n FROM _rv_r GROUP BY {keys})
        SELECT
            (SELECT COUNT(*) FROM _rv_l)                           AS left_rows,
            (SELECT COUNT(*) FROM _rv_r)                           AS right_rows,
            COALESCE(SUM(lc.n * rc.n), 0)                          AS matched_rows,
            COALESCE(SUM(lc.n) FILTER (WHERE rc.n IS NULL), 0)     AS left_unmatched,
            COALESCE(SUM(rc.n) FILTER (WHERE lc.n IS NULL), 0)     AS right_unmatched,
            COALESCE(MAX(rc.n) FILTER (WHERE lc.n IS NOT NULL), 0) AS max_right_per_key,
            COALESCE(MAX(lc.n) FILTER (WHERE rc.n IS NOT NULL), 0) AS max_left_per_key
        FROM lc FULL OUTER JOIN rc ON {on}
    """
    names = ["left_rows", "right_rows", "matched_rows", "left_unmatched",
             "right_unmatched", "max_right_per_key", "max_left_per_key"]
    return {n: int(v) for n, v in zip(names, con.execute(q).fetchone())}


# --------------------------------------------------------------------------
# Check 2: filters on nullable columns
# --------------------------------------------------------------------------
# WHY THIS IS A CORRECTNESS ISSUE, NOT A STYLE ISSUE (three-valued logic)
#
# SQL conditions have three results: TRUE, FALSE and UNKNOWN. Any comparison
# with NULL -- NULL = 'x', NULL <> 'x', NULL > 0, NULL IN (...) -- is UNKNOWN,
# and NOT UNKNOWN is still UNKNOWN. WHERE keeps only TRUE rows, so UNKNOWN rows
# are dropped exactly like FALSE rows, with no error and no warning.
#
# That changes the answer, not just how the query reads:
#   * `status <> 'refunded'` reads as "everything except refunds", but rows
#     with an unknown status are removed too, so a total built on it is short
#     by whatever those rows carried.
#   * After `customers LEFT JOIN orders`, customers with no orders have NULL
#     in every orders column. A WHERE on any orders column drops them, so the
#     LEFT JOIN silently behaves like an INNER JOIN.
# Two queries that look equivalent can return different numbers, which is why
# this is flagged with a row count rather than as a formatting note.
def _null_sensitive_columns(expr, negated=False):
    """Yield (column_ref, inside_a_negation) for columns whose NULL makes the predicate UNKNOWN."""
    if isinstance(expr, list):
        for e in expr:
            yield from _null_sensitive_columns(e, negated)
        return
    if not isinstance(expr, dict):
        return
    cls, typ = expr.get("class"), expr.get("type")
    if not isinstance(typ, str):   # CHANGED: constants store their data type as a dict under "type"
        typ = None
    if typ in _NULL_AWARE_TYPES or cls == "SUBQUERY":
        return
    if cls == "FUNCTION" and expr.get("function_name", "").lower() in _NULL_AWARE_FUNCS:
        return
    if cls == "COLUMN_REF":
        yield expr, negated
        return
    if typ == "CONJUNCTION_OR":
        # `col <> 'x' OR col IS NULL` keeps the unknowns on purpose.
        handled = {_colkey(n["children"][0]) for n in _walk(expr)
                   if n.get("type") == "OPERATOR_IS_NULL" and n.get("children")
                   and n["children"][0].get("class") == "COLUMN_REF"}
        for child in expr.get("children", []):
            for c, neg in _null_sensitive_columns(child, negated):
                if _colkey(c) not in handled:
                    yield c, neg
        return
    negated = negated or typ in _NEGATIONS
    for v in expr.values():
        if isinstance(v, (dict, list)):
            yield from _null_sensitive_columns(v, negated)


def _check_null_filter(con, root, r, negated, scope, nullable_side, where, out):
    alias, col = r
    name = f"{alias}.{col}"
    leaf_sql = _select_from(con, root, scope[alias]["ref"])
    nulls, total = con.execute(
        f"SELECT COUNT(*) FILTER (WHERE {_q(col)} IS NULL), COUNT(*) FROM ({leaf_sql})").fetchone()
    details = {"column": name, "null_rows": int(nulls), "rows": int(total), "negated": negated}

    if alias in nullable_side:
        label, unmatched = nullable_side[alias]
        msg = (f"WHERE filters on {name}, but {alias} is the optional side of {label}. "
               f"Rows with no match have {name} = NULL, the condition is UNKNOWN for them, "
               "and WHERE drops them, so the outer join quietly becomes an INNER JOIN")
        msg += f" ({unmatched:,} unmatched rows lost)." if unmatched else "."
        msg += f" Move the condition into ON, or add `OR {name} IS NULL`."
        out.append(QueryWarning("null_filter", "high", msg, where,
                                {**details, "outer_join": label, "unmatched_rows": unmatched}))
        return

    if nulls:
        msg = (f"{nulls:,} of {total:,} rows have {name} = NULL. The filter is UNKNOWN for them, "
               "so WHERE drops them without telling you.")
        if negated:
            msg += (" This is a NOT / <> / NOT IN filter, which reads as 'everything except X', "
                    "but the unknown rows are excluded too.")
        msg += f" Decide explicitly: add `OR {name} IS NULL` to keep them, or `AND {name} IS NOT NULL` to document dropping them."
        out.append(QueryWarning("null_filter", "high" if negated else "medium", msg, where, details))
    elif scope[alias]["columns"].get(col):
        out.append(QueryWarning("null_filter", "info",
                                f"{name} has no NULLs today but the column allows them; "
                                "this filter will start dropping rows when one appears.",
                                where, details))


# --------------------------------------------------------------------------
# Check 3: aggregates over a fanned-out result
# --------------------------------------------------------------------------
def _check_aggregate(agg, scope, duplicated, upstream, where, out):
    fname = agg.get("function_name", "").lower()
    if fname in _DUPLICATE_SAFE or agg.get("distinct"):
        return
    children = agg.get("children", [])
    if fname == "count_star" or (fname == "count" and all(c.get("class") == "STAR" for c in children)):
        if duplicated:
            out.append(QueryWarning(
                "fanout_aggregate", "info",
                f"COUNT(*) counts joined rows, so rows of {', '.join(sorted(duplicated))} are counted "
                "once per match. Use COUNT(DISTINCT <key>) if you meant to count those.", where))
        return

    hits = {}
    for c in (n for n in _walk(children) if n.get("class") == "COLUMN_REF"):
        r = _resolve(c, scope)
        if not r:
            continue
        a, col = r
        if a in duplicated:
            hits[f"{a}.{col}"] = (duplicated[a], "a join in this SELECT")
        elif a in upstream and col in upstream[a]["columns"]:
            hits[f"{a}.{col}"] = (upstream[a]["factor"], f"a join in {upstream[a]['source']}")
    if not hits:
        return
    factor = max(f for f, _ in hits.values())
    sources = sorted({s for _, s in hits.values()})
    effect = ("the average is weighted toward rows with more matches"
              if fname in ("avg", "mean") else "the result is inflated")
    out.append(QueryWarning(
        "fanout_aggregate", "high",
        f"{fname.upper()}() uses {', '.join(hits)}, but {' and '.join(sources)} repeats each of "
        f"those rows up to {factor}x, so {effect}. Aggregate that table before the join, "
        "or join to a version with one row per key.",
        where, {"function": fname, "columns": list(hits), "max_copies": factor}))
    
# CHANGED: removed the `if __name__ == "__main__":` toy self-check (in a notebook it runs and replaces `con` with toy data)
