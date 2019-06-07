import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

sz = 10

mean = []
X = []
Y = []
Z = []
pats = []

f = open("grid.log")

while True:
        line = f.readline()
        if line == "Sampled points dump:\n":
                break

while True:
        line = f.readline()
        if line == "Enter coordinates to test:\n":
                break
        patt,cnt = line.split()
        patt,cnt = int(patt),int(cnt)
        pats.append(patt)
        mean.append(tuple(map(int,f.readline().split())))
        pnt = [tuple(map(int,f.readline().split())) for i in range(cnt)]
        x_, y_, z_ = zip(*pnt)
        X.append(x_)
        Y.append(y_)
        Z.append(z_)

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(131, projection='3d')
for idx,(x,y,z,p) in enumerate(zip(X,Y,Z,pats)):
        ax.scatter(x,y,z,label=p,s=sz,c='C'+str(idx))
X,Y,Z = zip(*mean)
for x,y,z,p in zip(X,Y,Z,pats):
        ax.scatter(x,y,z,label=p,s=sz*10)

ax.legend(fontsize='small')

pnts1 = dict((pat, []) for pat in pats)
pnts2 = dict((pat, []) for pat in pats)
X = []
Y = []
Z = []

while True:
        line = f.readline()
        if not line:
                break
        l = line.split()
        act = int(l[-3])
        pat = int(l[-1])
        pnts1[pat].append(( int(l[0]), int(l[1]), int(l[2]) ))
        if pat != act:
                pnts2[pat].append(( int(l[0]), int(l[1]), int(l[2]) ))

for p in pats:
        x,y,z = zip(*pnts1[p]) if len(pnts1[p]) > 0 else ([],[],[])
        X.append(x)
        Y.append(y)
        Z.append(z)

ax = fig.add_subplot(132, projection='3d')
for idx,(x,y,z,p) in enumerate(zip(X,Y,Z,pats)):
        ax.scatter(x,y,z,label=p,s=sz,c='C'+str(idx))

ax.legend(fontsize='small')

X = []
Y = []
Z = []

for p in pats:
        x,y,z = zip(*pnts2[p]) if len(pnts2[p]) > 0 else ([],[],[])
        X.append(x)
        Y.append(y)
        Z.append(z)

ax = fig.add_subplot(133, projection='3d')
for idx,(x,y,z,p) in enumerate(zip(X,Y,Z,pats)):
        ax.scatter(x,y,z,label=p,s=sz,c='C'+str(idx))

ax.legend(fontsize='small')

plt.show()


