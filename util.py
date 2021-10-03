import numpy as np

def grid_topology_to_faces(rows, cols):
	F = []
	for i in range(rows - 1):
		for j in range(cols - 1):
			f = [i*cols + j, i*cols + j + 1, (i + 1)*cols + j + 1, (i + 1)*cols + j]
			F += [f]
	return F

def quads2tris(F):
	out = []
	for f in F:
		if len(f) == 3: out += [f]
		elif len(f) == 4: out += [[f[0], f[1], f[2]],
								  [f[0], f[2], f[3]]]
		else: sys.exit()
	return np.array(out, np.int32)