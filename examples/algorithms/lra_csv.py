import phylanx
from phylanx.util import phy_print

et = phylanx.execution_tree

file_name = "breast_cancer.csv"

data = et.file_read_csv(file_name)
x = et.slice(data, 0, 569, 0, 30)
y = et.slice(data, 0, 569, 30, 31)
num_iters = 750
alpha = et.var(1e-5)
gradient = et.zeros(x.dimension(1))
wieghts = et.zeros(x.dimension(1))
error = et.zeros(x.dimension(0))
pred = et.zeros(x.dimension(0))
transx = et.transpose(x)

for step in range(num_iters):
    g = et.dot(x, wieghts)
    dot = et.dot(x, wieghts)
    mul = et.multiply(et.var(-1), dot)
    e = et.exp(mul)
    add = et.add(et.var(1.0), e)
    pred = et.div(et.var(1.0), add)
    error = et.subtract(pred, y)
    gradient = et.dot(transx, error)
    mul = et.multiply(alpha, gradient)
    wieghts = et.subtract(wieghts, mul)

phy_print(wieghts)
