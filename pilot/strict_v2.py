"""`strict_v2_display_primary` -- score on the display span, demote SymPy to a warning.

The 2026-08-12 false-pass audit showed why this rule is needed. Of the 20
`strict_v1`-CORRECT items a human ruled unearned, **the failure is almost
always the SPAN, not the normalizer**: 6 were a one-character span the
extractor chose (item 84's span is literally `c`), 8 were a partial span
(item 27 keeps `4x = 3` and drops `y = 33/4`), and only 2 were SymPy
collapsing a long span to one symbol. Two collapsed labels then MATCH and the
item scores correct for no reason at all.

So this rule inverts the authority:

* **PRIMARY** -- the normalized display SPAN, model against truth. That is the
  text a human reads off the page, and it is what the audit adjudicated.
* **HELPER ONLY** -- SymPy. It never decides correctness here. It contributes
  `sympy_match` and three warnings, because a SymPy agreement on top of a span
  agreement is reassuring and a SymPy *dis*agreement is worth a second look.

Everything else is a RISK FLAG, not a verdict. A flag says "this is the shape
of item the automatic scorer gets wrong", so a human pass can be aimed at 30
items instead of 300. `tiny_valid_mcq` and `tiny_suspicious_non_mcq` exist as
separate flags for exactly this reason -- a one-letter span `B` is a perfectly
good answer to a multiple-choice item and near-worthless anywhere else, and
collapsing them into one "short answer" flag would hide that.

**Human visual reading overrides every label in this module**, and where the
display span is richer than the SymPy label, the span is what to believe.

NOTHING HERE TOUCHES THE FROZEN PIPELINE. `rescore.RULES` is unchanged and
`strict_v1` still produces every locked number; this is an alternative
reported alongside, in the same spirit as `pilot.rescore`'s other rules.
"""

import ast
import re
from typing import Optional, Sequence

import pandas as pd

from . import canonicalize, entropy, parsing, rescore

RULE_NAME = "strict_v2_display_primary"

#: Metadata SymPy is still allowed to contribute. None of these decide scoring.
SYMPY_FLAGS = ("sympy_match", "sympy_partial_parse_risk",
               "sympy_malformed_derivative", "multi_answer_collapse")

#: Answer-shape flags. Each marks a kind of item the automatic scorer is known
#: to mishandle; they are diagnostic, never a verdict.
RISK_FLAGS = ("mcq_option", "tiny_valid_mcq", "tiny_suspicious_non_mcq",
              "multi_value_answer", "set_answer", "system_answer",
              "derivative_equation", "text_conclusion")

ALL_FLAGS = SYMPY_FLAGS + RISK_FLAGS

_OPTION_RE = re.compile(r"\boption\b", re.I)
_BARE_OPTION_RE = re.compile(r"^\(?\s*(?:option\s*)?[a-eA-E]\s*\)?[.)]?$")
# Set notation, NOT "any LaTeX braces" -- \frac{a}{b} has braces and is not a
# set. A first version matched bare {...} and fired on 45% of items, which made
# the review queue useless. Detected on the RAW span, because normalization
# turns \{ into { and the distinction is then unrecoverable.
_SET_RE = re.compile(r"\\[{}]|\\(?:cup|cap|setminus|subset|supset|emptyset)\b"
                     r"|\\in\b|(?<![a-zA-Z])\{[^{}]*,[^{}]*\}")
_DERIV_RE = re.compile(r"\\frac\s*\{\s*d(?:\^?\d?)?\s*[a-z]?\s*\}|"
                       r"\bd\^?2?\s*y\s*/\s*d\s*x|\\mathrm\{d\}|"
                       r"derivative\(|\bdy\s*/\s*dx\b", re.I)
_MALFORMED_DERIV_RE = re.compile(r"d\*\*\d\s*\*\s*[a-z]|/\s*\(?\s*dx\s*\*\*|"
                                 r"\bdx\*\*\d", re.I)
# Words that mean the span is a sentence rather than a value. SymPy's own
# function names are excluded -- `eq`, `log`, `tan` are not prose, and treating
# them as prose is the exact over-trigger that once mislabelled item 55.
_SYMPY_WORDS = {"eq", "log", "tan", "sin", "cos", "cot", "sec", "csc", "exp",
                "sqrt", "abs", "integral", "derivative", "limit", "sum", "pi",
                "oo", "re", "im", "conjugate", "matrix", "and", "or", "true",
                "false"}
_WORD_RE = re.compile(r"[A-Za-z]{3,}")

_SPACING_MACROS = re.compile(r"\\(?:,|;|:|!|quad|qquad|ensuremath|displaystyle"
                             r"|left|right|bigl|bigr|hspace\{[^}]*\})")


def normalize_span(span: Optional[str]) -> str:
    """Cosmetic normalization of a display span, and nothing more.

    Deliberately conservative: it removes LaTeX spacing and wrapper macros,
    unifies bracket and multiplication spellings, drops trailing punctuation
    and collapses whitespace. It does NOT reorder terms, evaluate arithmetic
    or canonicalize equations -- those are exactly the operations that let two
    different answers land on the same label, which is what this rule exists
    to stop.
    """
    if span is None:
        return ""
    s = str(span)
    for macro in ("textcolor", "text", "mathrm", "mathbf", "textbf", "mbox",
                  "operatorname"):
        s = canonicalize.unwrap_latex_macro(s, macro)
    s = _SPACING_MACROS.sub(" ", s)
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = re.sub(r"\\[a-zA-Z]+\s*", lambda m: m.group(0).strip() + " ", s)
    s = s.replace("$", "").replace("\\\\", " ")
    s = re.sub(r"[ \t\n\r]+", " ", s)
    s = s.strip().rstrip(".;,")
    s = re.sub(r"\s*([=+\-*/,])\s*", r"\1", s)
    return s.lower().strip()


def _looks_mcq(question: Optional[str], gt_span: Optional[str],
               gt_field: Optional[str]) -> bool:
    blob = " ".join(str(x) for x in (question, gt_span, gt_field) if x)
    if _OPTION_RE.search(blob):
        return True
    return bool(re.search(r"\(\s*[a-d]\s*\)[^\n]{0,60}\(\s*[b-e]\s*\)", blob, re.I))


def _is_prose(span: str) -> bool:
    words = [w.lower() for w in _WORD_RE.findall(span)]
    real = [w for w in words if w not in _SYMPY_WORDS]
    return len(real) >= 3


def answer_flags(span: str, sympy_label: str, is_mcq: bool) -> dict:
    """Answer-shape and SymPy-risk flags for one side of a comparison."""
    norm = normalize_span(span)
    bare = norm.strip()
    n_eq = len(re.findall(r"=", bare))
    # A SYSTEM is several separate equations, not one chain. `a=b=c` has two
    # "=" and is a single statement; `x=3, y=1` is two. Counting raw "=" made
    # this fire on 31% of items and drowned the queue.
    parts = [p for p in re.split(r",|;|\band\b|\\quad", bare) if p.strip()]
    n_eq_parts = sum(1 for p in parts if "=" in p)
    commas = [p for p in re.split(r",", bare) if p.strip()]
    payload = str(sympy_label).split(":", 1)[-1] if sympy_label else ""

    tiny = len(bare) <= 3
    # A bare letter is only an "option answer" when the item actually IS
    # multiple-choice. Without that guard item 289's set-equality answer `B`
    # reads as an option and the tiny-span warning it needs gets contradicted.
    mcq_option = bool(_OPTION_RE.search(bare)) or (
        is_mcq and bool(_BARE_OPTION_RE.match(bare)))
    multi_value = len(commas) >= 2 and n_eq == 0 and len(bare) > 3
    system = n_eq_parts >= 2

    return {
        "mcq_option": mcq_option,
        "tiny_valid_mcq": tiny and is_mcq,
        "tiny_suspicious_non_mcq": tiny and not is_mcq,
        "multi_value_answer": multi_value,
        "set_answer": bool(_SET_RE.search(str(span))),
        "system_answer": system,
        "derivative_equation": bool(_DERIV_RE.search(str(span))),
        "text_conclusion": _is_prose(bare),
        # SymPy risk: the label carries strictly less than the span did.
        "sympy_partial_parse_risk": bool(
            payload and system and payload.count("eq(") <= 1),
        "sympy_malformed_derivative": bool(
            _MALFORMED_DERIV_RE.search(payload)),
        "multi_answer_collapse": bool(
            payload and multi_value and len(payload) <= max(3, len(bare) // 3)),
    }


def score_item_v2(raw_samples: Sequence[str], ground_truth: Optional[str],
                  question: Optional[str] = None) -> dict:
    """Score one item on the display span, with SymPy demoted to metadata."""
    v1 = rescore.trace_item(raw_samples, ground_truth, "strict_v1")

    gt_field = v1["ground_truth"]["answer_field"]
    gt_span = v1["ground_truth"]["final_answer"]
    gt_norm = normalize_span(gt_span)
    is_mcq = _looks_mcq(question, gt_span, gt_field)

    spans = [s["final_answer"] for s in v1["samples"]]
    norms = [normalize_span(s) if s is not None else canonicalize.PARSE_FAILURE_SENTINEL
             for s in spans]
    majority, count = entropy.majority_cluster(norms)
    rep = next((i for i, n in enumerate(norms) if n == majority), 0)

    correct = bool(majority) and majority == gt_norm and gt_norm != ""
    flags = answer_flags(spans[rep] or "", v1["samples"][rep]["label"], is_mcq)
    gt_flags = answer_flags(gt_span or "", v1["ground_truth"]["label"], is_mcq)
    # A risk present on EITHER side is a risk for the comparison.
    for k in RISK_FLAGS + ("sympy_partial_parse_risk",
                           "sympy_malformed_derivative", "multi_answer_collapse"):
        flags[k] = bool(flags.get(k)) or bool(gt_flags.get(k))

    return {
        "correct_strict_v1": v1["correct"],
        "correct_strict_v2_display_primary": correct,
        "span_entropy": entropy.cluster_entropy(norms),
        "perception_entropy": v1["perception_entropy"],
        "model_span": spans[rep],
        "truth_span": gt_span,
        "model_span_norm": majority,
        "truth_span_norm": gt_norm,
        "model_label": v1["samples"][rep]["label"],
        "truth_label": v1["ground_truth"]["label"],
        "model_tier": v1["samples"][rep]["tier"],
        "truth_tier": v1["ground_truth"]["tier"],
        "is_mcq": is_mcq,
        "sympy_match": v1["correct"],
        **{k: bool(flags[k]) for k in ALL_FLAGS if k != "sympy_match"},
    }


def rescore_v2(df: pd.DataFrame,
               samples_col: str = "all_transcription_samples_raw",
               gt_col: str = "pert_a", question_col: str = "orig_q",
               progress: bool = False) -> pd.DataFrame:
    """Apply `strict_v2_display_primary` to a whole run. Input never mutated."""
    rows = []
    it = df.index
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=RULE_NAME)
        except ImportError:
            pass
    for i in it:
        rows.append(score_item_v2(
            ast.literal_eval(df.loc[i, samples_col]), df.loc[i, gt_col],
            df.loc[i, question_col] if question_col in df.columns else None))
    return pd.DataFrame(rows, index=df.index)


def disagreement_and_risk_sheet(scored: pd.DataFrame,
                                path: Optional[str] = None) -> pd.DataFrame:
    """Only the items worth a human look: rules disagree, OR a risk flag fires.

    The point of the whole exercise -- turn "audit 300 items" into "audit the
    ones the two rules cannot agree on plus the ones whose shape is known to
    break the scorer". Ships with an empty `final_label` column, the same
    confirm-or-correct shape as every other coding sheet here.
    """
    risky = scored[list(RISK_FLAGS)].any(axis=1)
    sympy_risky = scored[["sympy_partial_parse_risk",
                          "sympy_malformed_derivative",
                          "multi_answer_collapse"]].any(axis=1)
    disagree = (scored["correct_strict_v1"]
                != scored["correct_strict_v2_display_primary"])
    sel = scored[disagree | risky | sympy_risky].copy()
    sel.insert(0, "item", sel.index)
    sel["rules_disagree"] = disagree[sel.index]
    # MEASURED, not assumed: the full "any flag" queue is 71% of the run and
    # catches 15 of 20 known false passes, which is what a RANDOM 71% would
    # catch. The high tier is 36% of the run and catches the same 15 -- 2.1x
    # enrichment. Sort by it or the queue is no better than reading everything.
    sel["priority"] = review_priority(scored)[sel.index]
    sel = sel.sort_values(
        ["priority", "item"],
        key=lambda c: c.map({"high": 0, "medium": 1, "low": 2}) if c.name == "priority" else c)
    sel["final_label"] = ""
    sel["confidence"] = ""
    sel["note"] = ""
    if path:
        sel.to_csv(path, index=False)
    return sel


def accuracy_summary(scored: pd.DataFrame) -> dict:
    """v1 vs v2 accuracy, the disagreement split, and the flag counts."""
    v1 = scored["correct_strict_v1"].astype(bool)
    v2 = scored["correct_strict_v2_display_primary"].astype(bool)
    dis = v1 != v2
    return {
        "n": len(scored),
        "strict_v1_correct": int(v1.sum()),
        "strict_v1_accuracy": float(v1.mean()),
        "strict_v2_correct": int(v2.sum()),
        "strict_v2_accuracy": float(v2.mean()),
        "n_disagree": int(dis.sum()),
        "v1_only_correct": int((v1 & ~v2).sum()),
        "v2_only_correct": int((~v1 & v2).sum()),
        "flag_counts": {k: int(scored[k].astype(bool).sum())
                        for k in ALL_FLAGS if k in scored},
    }


#: Flags that actually predict a false pass, versus flags that merely describe
#: the item's shape. `mcq_option` is descriptive -- 52 of the 300 items are
#: multiple-choice and most are scored fine -- so putting it in the queue at
#: equal weight buries the signal.
HIGH_PRIORITY_FLAGS = ("tiny_suspicious_non_mcq", "multi_answer_collapse",
                       "sympy_partial_parse_risk", "sympy_malformed_derivative")


def review_priority(scored: pd.DataFrame) -> pd.Series:
    """Triage order for the review queue.

    Exists because "any risk flag" selects ~70% of the run, which is no better
    than reading everything. Measured enrichment for false passes is what
    decides the tiers, not intuition.
    """
    disagree = (scored["correct_strict_v1"]
                != scored["correct_strict_v2_display_primary"])
    high = disagree | scored[list(HIGH_PRIORITY_FLAGS)].any(axis=1)
    medium = scored[["multi_value_answer", "set_answer", "system_answer",
                     "derivative_equation", "text_conclusion"]].any(axis=1)
    out = pd.Series("low", index=scored.index)
    out[medium] = "medium"
    out[high] = "high"
    return out


#: Sort for the human audit sheet: the items most likely to be a false pass
#: first, then a stable tiebreak on item id so two builds never differ.
AUDIT_SORT = ("rules_disagree", "tiny_suspicious_non_mcq",
              "sympy_partial_parse_risk", "multi_answer_collapse")

_HTML_HEAD = """<meta charset="utf-8">
<title>strict_v2 high-priority human audit</title>
<style>
:root{--bg:#fbfaf7;--fg:#22201d;--mut:#6b6560;--line:#e2ddd5;--card:#fff;
      --warn:#8a5a00;--warnbg:#fdf3e0;--dis:#8a2f2f;--disbg:#fdeaea;--code:#f3f0ea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#191817;--fg:#eae6e0;--mut:#a49d95;--line:#332f2b;--card:#211f1d;
  --warn:#e0aa5a;--warnbg:#332813;--dis:#e08a8a;--disbg:#331a1a;--code:#2a2724}}
:root[data-theme="dark"]{--bg:#191817;--fg:#eae6e0;--mut:#a49d95;--line:#332f2b;
  --card:#211f1d;--warn:#e0aa5a;--warnbg:#332813;--dis:#e08a8a;--disbg:#331a1a;--code:#2a2724}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:24px;
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{max-width:1000px;margin:0 auto 28px}
h1{font-size:22px;margin:0 0 8px}
.sub{color:var(--mut);font-size:14px;max-width:70ch}
.card{max-width:1000px;margin:0 auto 20px;background:var(--card);
 border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card.dis{border-left:4px solid var(--dis)}
.hd{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;margin-bottom:10px}
.id{font-weight:650;font-size:17px}
.meta{color:var(--mut);font-size:13px}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}
.chip{font-size:11px;padding:2px 8px;border-radius:999px;
 background:var(--warnbg);color:var(--warn);border:1px solid transparent}
.chip.d{background:var(--disbg);color:var(--dis)}
figure{margin:12px 0}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;
 background:#fff;display:block}
.noimg{padding:22px;border:1px dashed var(--line);border-radius:6px;
 color:var(--mut);font-size:13px;text-align:center}
table.kv{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}
table.kv td{padding:5px 8px;border-top:1px solid var(--line);vertical-align:top}
table.kv td:first-child{color:var(--mut);white-space:nowrap;width:1%}
code{background:var(--code);padding:1px 5px;border-radius:4px;
 font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-word}
.fields{display:grid;grid-template-columns:200px 130px 1fr;gap:8px;margin-top:12px}
@media(max-width:720px){.fields{grid-template-columns:1fr}}
select,input,textarea{font:inherit;font-size:13px;padding:6px 8px;
 border:1px solid var(--line);border-radius:6px;background:var(--bg);
 color:var(--fg);width:100%}
textarea{min-height:38px;resize:vertical}
.bar{position:sticky;top:0;z-index:9;background:var(--bg);padding:10px 0 14px;
 border-bottom:1px solid var(--line);margin-bottom:20px}
.bar .in{max-width:1000px;margin:0 auto;display:flex;gap:10px;align-items:center;
 flex-wrap:wrap}
button{font:inherit;font-size:13px;padding:7px 14px;border-radius:6px;
 border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--mut)}
.count{color:var(--mut);font-size:13px}
</style>
"""


def _chip(name: str, dis: bool = False) -> str:
    return f'<span class="chip{" d" if dis else ""}">{name}</span>'


def high_priority_audit_sheet(queue: pd.DataFrame, out_csv: str, out_html: str,
                              image_dir: str = "images") -> pd.DataFrame:
    """Build the CSV + HTML coding sheet for the high-priority tier.

    `image_path` is RELATIVE to the HTML, because the FERMAT images are gated
    and only exist in Colab -- notebook 23 exports them next to these files.
    The HTML degrades to a labelled placeholder rather than a broken image
    when they are not there yet, so the sheet is readable either way.

    The human fields ship empty and are editable in the browser; the page
    keeps them in localStorage and can emit the finished CSV, so a pass can be
    done in one sitting without hand-editing a spreadsheet. **Nothing here
    scores anything** -- human reading overrides every automatic label.
    """
    high = queue[queue["priority"] == "high"].copy()
    high["item_id"] = high["item"].astype(int)
    for c in AUDIT_SORT:
        high[c] = high[c].astype(bool)
    high = high.sort_values(list(AUDIT_SORT) + ["item_id"],
                            ascending=[False] * len(AUDIT_SORT) + [True])
    high["image_path"] = high["item_id"].map(
        lambda i: f"{image_dir}/item{i:03d}.png")

    flag_cols = [c for c in ALL_FLAGS if c in high.columns and c != "sympy_match"]
    cols = (["item_id", "image_path", "priority",
             "span_m_disp", "span_t_disp", "label_m", "label_t",
             "strict_v1_correct", "strict_v2_correct", "rules_disagree"]
            + flag_cols + ["final_label", "confidence", "note"])
    out = high.rename(columns={
        "model_span": "span_m_disp", "truth_span": "span_t_disp",
        "model_label": "label_m", "truth_label": "label_t",
        "correct_strict_v1": "strict_v1_correct",
        "correct_strict_v2_display_primary": "strict_v2_correct"})
    out = out.reindex(columns=cols)
    out.to_csv(out_csv, index=False)

    import html as _h
    parts = [_HTML_HEAD,
             '<div class="bar"><div class="in">',
             '<button onclick="exportCsv()">Export CSV</button>',
             '<button onclick="clearAll()">Clear</button>',
             f'<span class="count" id="cnt">0 / {len(out)} coded</span>',
             '</div></div>',
             '<header><h1>strict_v2 high-priority human audit</h1>',
             f'<p class="sub">{len(out)} items, the tier measured to enrich for '
             'false passes 2.1&times; over chance. <strong>Your reading of the '
             'image overrides every automatic label below.</strong> Where the '
             'display span is richer than the SymPy label, believe the span. '
             'Labels: <code>true_correct</code>, <code>extraction_issue</code>, '
             '<code>needs_visual</code>.</p></header>']

    for _, r in out.iterrows():
        i = int(r["item_id"])
        chips = ([_chip("rules disagree", True)] if r["rules_disagree"] else [])
        chips += [_chip(c.replace("_", " ")) for c in flag_cols if bool(r[c])]
        parts.append(
            f'<div class="card{" dis" if r["rules_disagree"] else ""}">'
            f'<div class="hd"><span class="id">item {i}</span>'
            f'<span class="meta">strict_v1={"correct" if r["strict_v1_correct"] else "wrong"}'
            f' &middot; strict_v2={"correct" if r["strict_v2_correct"] else "wrong"}</span></div>'
            f'<div class="chips">{"".join(chips)}</div>'
            f'<figure><img src="{_h.escape(str(r["image_path"]))}" alt="item {i}" '
            f'loading="lazy" onerror="this.outerHTML='
            f'\'&lt;div class=&quot;noimg&quot;&gt;image not exported yet - run notebook 23&lt;/div&gt;\'">'
            f'</figure>'
            '<table class="kv">'
            f'<tr><td>span M</td><td><code>{_h.escape(str(r["span_m_disp"]))}</code></td></tr>'
            f'<tr><td>span T</td><td><code>{_h.escape(str(r["span_t_disp"]))}</code></td></tr>'
            f'<tr><td>label M</td><td><code>{_h.escape(str(r["label_m"]))}</code></td></tr>'
            f'<tr><td>label T</td><td><code>{_h.escape(str(r["label_t"]))}</code></td></tr>'
            '</table>'
            f'<div class="fields">'
            f'<select data-i="{i}" data-f="final_label"><option value=""></option>'
            '<option>true_correct</option><option>extraction_issue</option>'
            '<option>needs_visual</option></select>'
            f'<select data-i="{i}" data-f="confidence"><option value=""></option>'
            '<option>high</option><option>medium</option><option>low</option></select>'
            f'<textarea data-i="{i}" data-f="note" placeholder="note"></textarea>'
            '</div></div>')

    parts.append("""<script>
const KEY='strict_v2_audit_20260812';
const S=JSON.parse(localStorage.getItem(KEY)||'{}');
const els=[...document.querySelectorAll('[data-i]')];
function count(){const n=new Set(els.filter(e=>e.dataset.f==='final_label'&&e.value)
  .map(e=>e.dataset.i)).size;document.getElementById('cnt').textContent=
  n+' / '+document.querySelectorAll('.card').length+' coded';}
els.forEach(e=>{const k=e.dataset.i+'|'+e.dataset.f;if(S[k])e.value=S[k];
  e.addEventListener('input',()=>{S[k]=e.value;
    localStorage.setItem(KEY,JSON.stringify(S));count();});});
count();
function q(s){s=(s==null?'':String(s));return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function exportCsv(){const ids=[...new Set(els.map(e=>e.dataset.i))];
  let out='item_id,final_label,confidence,note\\n';
  ids.forEach(i=>{const g=f=>(S[i+'|'+f]||'');
    out+=[i,q(g('final_label')),q(g('confidence')),q(g('note'))].join(',')+'\\n';});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([out],{type:'text/csv'}));
  a.download='strict_v2_high_priority_coded.csv';a.click();}
function clearAll(){if(!confirm('Clear all entered labels?'))return;
  localStorage.removeItem(KEY);location.reload();}
</script>""")
    with open(out_html, "w") as fh:
        fh.write("\n".join(parts))
    return out
