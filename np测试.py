import numpy as np
import taichi as ti
import random

ti.init()


_offsets = np.array([
[[0, -1], [1, 0], [0, -2]],
[[1, 1], [-1, 0], [1, 0]],
[[0, -1], [-1, 0], [0, -2]],
[[0, 1], [1, 0], [1, -1]],
[[1, 0], [2, 0], [-1, 0]],
[[0, 1], [1, 1], [1, 0]],
[[-1, 0], [1, 0], [0, 1]],
])
begin=1
# print(_offsets[begin:3])


xx=None
# print(x)

@ti.kernel
def ceshi():
    for i in range(10):
        print([random.random(),random.random()])
# ceshi()





dim=2
xx = ti.Vector.field(dim, dtype=ti.f32)
v = ti.Vector.field(dim, dtype=ti.f32)
C = ti.Matrix.field(dim, dim, dtype=ti.f32)
F = ti.Matrix.field(dim, dim, dtype=ti.f32)

material = ti.field(dtype=ti.i32)
color = ti.field(dtype=ti.i32)
Jp = ti.field(dtype=ti.f32)


# ti.root.dynamic(ti.i, 4).place(x, v, C, F, material,color, Jp)
# particle=ti.root.dynamic(ti.i, 4).place(x, v, C, F, material,color, Jp)
particle=ti.root.dynamic(ti.i, 4)

particle.place(xx, v, C, F, material,color, Jp)



offset = tuple(4096 // 2 for _ in range(2))
print(offset)


cs = ti.Vector.field(3, float, 10)


for i in range(10):
    cs[i]=ti.Vector([i,i,i])


def T(a):

    phi, theta = np.radians(0), np.radians(90)

    print(phi)
    print(theta)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)

    x, z = x * c + z * s, z * c - x * s
    
    u, v = x, y * C + z * S

    # return np.array([u, v]).swapaxes(0, 1) + 0.5
    return np.array([u, v]).swapaxes(0, 1) + 0.5


pos=cs.to_numpy()


for i in range(10):
    print(cs[i])

css=T(pos)
print("---------")


for i in range(10):
    print(css[i])


