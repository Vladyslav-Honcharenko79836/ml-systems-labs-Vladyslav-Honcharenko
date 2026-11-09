import math


class Value:
    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = _noop
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant int/float exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            ex = math.exp(x)
            s = ex / (1.0 + ex)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def zero_grad_graph(self):
        stack, seen = [self], set()
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            n.grad = 0.0
            stack.extend(n._prev)

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}{', op=' + self._op if self._op else ''})"


def _noop():
    pass


def vsum(values):
    values = list(values)
    out = values[0]
    for v in values[1:]:
        out = out + v
    return out


def print_graph(root):
    lines, seen = [], set()

    def walk(n, depth):
        tag = n.label or n._op or "leaf"
        lines.append("  " * depth + f"[{tag}] data={n.data:.4g} grad={n.grad:.4g}")
        if id(n) in seen:
            return
        seen.add(id(n))
        for p in n._prev:
            walk(p, depth + 1)

    walk(root, 0)
    return "\n".join(lines)
