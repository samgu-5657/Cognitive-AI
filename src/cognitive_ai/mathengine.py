"""Internal math engine (Phase 4C, domain A). Hand-implemented rules.

Own expression data structures (Num/Var/Add/Mul/Pow), own tokenizer +
recursive-descent parser for a RESTRICTED grammar, own rewrite rules.
No eval(), no external solver, no hardcoded answers: every result comes
with the rule applications that produced it.

Scope (explicit): integer/rational arithmetic (+-*/^, parentheses),
linear equations ax+b=c, polynomial differentiation (power/sum rules).
Anything else -> structured failure, never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


# ---- expressions ----
@dataclass(frozen=True)
class Num:
    v: Fraction

    def __str__(self):
        return str(self.v.numerator) if self.v.denominator == 1 \
            else f"{self.v.numerator}/{self.v.denominator}"


@dataclass(frozen=True)
class Var:
    name: str = "x"

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class Add:
    terms: tuple

    def __str__(self):
        return "+".join(str(t) for t in self.terms)


@dataclass(frozen=True)
class Mul:
    factors: tuple

    def __str__(self):
        return "*".join(str(f) for f in self.factors)


@dataclass(frozen=True)
class Pow:
    base: object
    exp: int

    def __str__(self):
        return f"{self.base}^{self.exp}"


class MathError(ValueError):
    pass


# ---- tokenizer + parser (restricted grammar, no eval) ----
def tokenize(s: str) -> list:
    toks, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            toks.append(("num", int(s[i:j])))
            i = j
        elif c.isalpha() or c in "xy":
            toks.append(("var", c))
            i += 1
        elif c in "+-*/^()=×÷":
            toks.append(("op", "×" if c == "×" else ("÷" if c == "÷" else c)))
            i += 1
        else:
            raise MathError(f"非法字符: {c!r}")
    return toks


class _Parser:
    def __init__(self, toks):
        self.toks = toks
        self.pos = 0

    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else (None, None)

    def eat(self, kind=None, val=None):
        t = self.peek()
        if t[0] is None:
            raise MathError("表达式意外结束")
        if kind and t[0] != kind:
            raise MathError(f"期望{kind},得到{t}")
        if val is not None and t[1] != val:
            raise MathError(f"期望{val},得到{t}")
        self.pos += 1
        return t

    def parse_expr(self):
        node = self.parse_term()
        terms = [node]
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.eat()[1]
            rhs = self.parse_term()
            terms.append(rhs if op == "+" else Mul((Num(Fraction(-1)), rhs)))
        return terms[0] if len(terms) == 1 else Add(tuple(terms))

    def parse_term(self):
        node = self.parse_factor()
        facs = [node]
        while self.peek() in (("op", "*"), ("op", "×"), ("op", "/"),
                              ("op", "÷")):
            op = self.eat()[1]
            rhs = self.parse_factor()
            if op in ("*", "×"):
                facs.append(rhs)
            else:
                facs.append(Pow(rhs, -1))
        # implicit multiplication: 2x, 2(x+1)
        while self.peek()[0] in ("num", "var") or self.peek() == ("op", "("):
            facs.append(self.parse_factor())
        return facs[0] if len(facs) == 1 else Mul(tuple(facs))

    def parse_factor(self):
        t = self.peek()
        if t == ("op", "("):
            self.eat()
            node = self.parse_expr()
            self.eat("op", ")")
        elif t[0] == "num":
            node = Num(Fraction(self.eat()[1]))
        elif t[0] == "var":
            node = Var(self.eat()[1])
        elif t == ("op", "-"):
            self.eat()
            node = Mul((Num(Fraction(-1)), self.parse_factor()))
        else:
            raise MathError(f"无法解析: {t}")
        if self.peek() == ("op", "^"):
            self.eat()
            e = self.eat("num")[1]
            node = Pow(node, e)
        return node


def parse_expr(s: str):
    p = _Parser(tokenize(s))
    node = p.parse_expr()
    if p.pos != len(p.toks):
        raise MathError(f"多余符号: {p.toks[p.pos:]}")
    return node


# ---- evaluation (exact rational arithmetic) ----
def evaluate(node, env: dict | None = None) -> Fraction:
    env = env or {}
    if isinstance(node, Num):
        return node.v
    if isinstance(node, Var):
        if node.name not in env:
            raise MathError(f"变量{node.name}未赋值")
        return env[node.name]
    if isinstance(node, Add):
        return sum((evaluate(t, env) for t in node.terms), Fraction(0))
    if isinstance(node, Mul):
        r = Fraction(1)
        for f in node.factors:
            r *= evaluate(f, env)
        return r
    if isinstance(node, Pow):
        b = evaluate(node.base, env)
        if node.exp < 0 and b == 0:
            raise MathError("除数为零")
        return b ** node.exp
    raise MathError(f"未知节点: {node}")


# ---- differentiation with recorded rule applications ----
def differentiate(node, var: str = "x") -> tuple:
    """Returns (derivative_expr, steps[{rule, before, after}])."""
    if isinstance(node, Num):
        return Num(Fraction(0)), [{"rule": "常数求导",
                                   "before": str(node), "after": "0"}]
    if isinstance(node, Var):
        d = Num(Fraction(1)) if node.name == var else Num(Fraction(0))
        return d, [{"rule": "变量求导" if node.name == var else "无关变量求导",
                    "before": str(node), "after": str(d)}]
    if isinstance(node, Add):
        ds, steps = [], [{"rule": "加法求导(逐项)",
                          "before": str(node), "after": ""}]
        for t in node.terms:
            d, s = differentiate(t, var)
            ds.append(d)
            steps.extend(s)
        out = ds[0] if len(ds) == 1 else Add(tuple(ds))
        steps[0]["after"] = str(out)
        return out, steps
    if isinstance(node, Mul):
        # product rule for exactly 2 factors; longer products fail loudly
        if len(node.factors) != 2:
            raise MathError("乘法求导仅支持两因子(其余请先展开)")
        f, g = node.factors
        df, sf = differentiate(f, var)
        dg, sg = differentiate(g, var)
        out = Add((Mul((df, g)), Mul((f, dg))))
        return out, [{"rule": "乘法求导(uv'=u'v+uv')",
                      "before": str(node), "after": str(out)}] + sf + sg
    if isinstance(node, Pow):
        if not isinstance(node.base, Var) or node.base.name != var:
            raise MathError("幂求导仅支持 x^n 形式")
        out = Mul((Num(Fraction(node.exp)), Pow(node.base, node.exp - 1))) \
            if node.exp != 1 else Num(Fraction(1))
        return out, [{"rule": "幂求导(nx^(n-1))",
                      "before": str(node), "after": str(out)}]
    raise MathError(f"无法求导: {node}")


def simplify_text(node) -> str:
    return str(node)


# ---- linear equations ----
def solve_linear(lhs, rhs, var: str = "x") -> tuple:
    """Solve lhs=rhs for var. Returns (solution Fraction, steps)."""
    steps = [{"rule": "移项", "before": f"{lhs}={rhs}", "after": ""}]
    # collect: lhs-rhs as a*x + b via probing (linearity check included)
    diff = Add((lhs, Mul((Num(Fraction(-1)), rhs))))
    f0 = evaluate(diff, {var: Fraction(0)})
    f1 = evaluate(diff, {var: Fraction(1)})
    a, b = f1 - f0, f0
    # linearity check at x=2
    if evaluate(diff, {var: Fraction(2)}) != 2 * a + b:
        raise MathError("非线性方程超出范围")
    if a == 0:
        raise MathError("无解或无穷解(非常数方程失效)")
    sol = -b / a
    steps[0]["after"] = f"{var}={sol}"
    steps.append({"rule": "回代验证", "before": f"{var}={sol}",
                  "after": f"lhs-rhs={evaluate(diff, {var: sol})}"})
    if evaluate(diff, {var: sol}) != 0:
        raise MathError("验证失败")
    return sol, steps


def split_equation(s: str):
    if s.count("=") != 1:
        raise MathError("方程必须恰含一个等号")
    l, r = s.split("=")
    return parse_expr(l.strip()), parse_expr(r.strip())
