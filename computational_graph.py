from math import log, e as E

def classify(e): return e if isinstance(e, Expression) else Number(e)
class Expression:
    def __add__(self, other): return Add(self, other)
    def __radd__(self, other): return Add(other, self)
    def __sub__(self, other): return Subtract(self, other)
    def __rsub__(self, other): return Subtract(other, self)
    def __mul__(self, other): return Multiply(self, other)
    def __rmul__(self, other): return Multiply(other, self)
    def __truediv__(self, other): return Divide(self, other)
    def __rtruediv__(self, other): return Divide(other, self)
    def __pow__(self, other): return Power(self, other)
    def __rpow__(self, other): return Power(other, self)
    def __abs__(self): return Absolute(self)
    def __neg__(self): return Negate(self)
    def __pos__(self): return self
    def __float__(self): self.f(); return self._
class Monad(Expression):
    def __init__(self, e): self.e = classify(e); self.f()
class Dyad(Expression):
    def __init__(self, l, r): self.l, self.r = classify(l), classify(r); self.f()
class Variable(Expression):
    def __init__(self, name, _=0): self.name, self._ = name, float(_)
    def assign(self, value): self._ = float(value)
    def f(self): pass
    def d(self, x): return int(x is self)
    def __eq__(self, other): self._ = float(other); return True
    def __repr__(self): return str(self.name)
class Number(Expression):
    def __init__(self, n): self._ = float(n)
    def f(self): pass
    def d(self, x): return 0
    def __repr__(self): return str(self._)
class Add(Dyad):
    def f(self): self._ = float(self.l) + float(self.r)
    def d(self, x): return self.l.d(x) + self.r.d(x)
    def __repr__(self): return f"({self.l} + {self.r})"
class Subtract(Dyad):
    def f(self): self._ = float(self.l) - float(self.r)
    def d(self, x): return self.l.d(x) - self.r.d(x)
    def __repr__(self): return f"({self.l} - {self.r})"
class Multiply(Dyad):
    def f(self): self._ = float(self.l) * float(self.r)
    def d(self, x): 
        return self.l.d(x) * float(self.r) + float(self.l) * self.r.d(x)
    def __repr__(self): return f"({self.l} * {self.r})"
class Divide(Dyad):
    def f(self): self._ = float(self.l) / float(self.r)
    def d(self, x): R = float(self.r); return (self.l.d(x) * R - float(self.l) * self.r.d(x)) / R ** 2
    def __repr__(self): return f"({self.l} / {self.r})"
class Power(Dyad):
    def f(self): self._ = float(self.l) ** float(self.r)
    def d(self, x):
        L, R = float(self.l), float(self.r)
        if L == 0: return 0
        if self.r.d(x) == 0: return R * L ** (R - 1) * self.l.d(x)
        return R * L ** (R - 1) * self.l.d(x) + L ** R * log(L) * self.r.d(x)
    def __repr__(self): return f"({self.l} ** {self.r})"
class Absolute(Monad):
    def f(self): self._ = abs(float(self.e))
    def d(self, x): return self.e.d(x) * (1 if float(self.e) >= 0 else -1)
    def __repr__(self): return f"|{self.e}|"
class Negate(Monad):
    def f(self): self._ = -float(self.e)
    def d(self, x): return -self.e.d(x)
    def __repr__(self): return f"-{self.e}"
class Log(Monad):
    def f(self): self._ = log(float(self.e))
    def d(self, x): return self.e.d(x) / float(self.e)
    def __repr__(self): return f"log({self.e})"
def Sqrt(x): return x ** 0.5
def Sigmoid(x): return 1 / (1 + E ** -x)
def ReLU(x): return (Absolute(x) + x) / 2

if __name__ == "__main__":
    x = Variable("x")
    y = Variable("y")
    x == 0
    y == 0
    f = Sqrt(x*x + y*y)
    print(f"f(x,y) = {float(f):.3f}")
    print(f"df/dx = {f.d(x):.3f}")
    print(f"df/dy = {f.d(y):.3f}")
    x == 1
    y == 1
    print(f"f(x,y) = {float(f):.3f}")
    print(f"df/dx = {f.d(x):.3f}")
    print(f"df/dy = {f.d(y):.3f}")
    x == 3
    y == 4
    print(f"f(x,y) = {float(f):.3f}")
    print(f"df/dx = {f.d(x):.3f}")
    print(f"df/dy = {f.d(y):.3f}")
