class Value:
    def __init__(self, data, _child=() ,_op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_child)
        self._backward= lambda: None
        self._op = _op
        

    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        output = Value(self.data * other.data, (self, other), "*")  

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        return output 

    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        output = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward   
        return output

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"ERROR : only supporting int/float powers"  
        output=Value(self.data**other,(self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * output.grad
        output._backward = _backward
        return output  

    def relu(self):
        output = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward
        return output   

    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

        