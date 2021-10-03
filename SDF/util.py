import sys
import numpy as np
	
def model_summary(targets):
	print("")
	_print = lambda x: print('\t' + x)
	sep = '---------------------------'
	total = 0
	_print(sep)
	_print('MODEL SUMMARY')
	_print(sep)
	for t in targets:
		_print(t.name + '\t' + str(t.shape))
		total += np.prod(t.shape)
	_print(sep)
	_print('Total params: ' + str(total))
	_print(sep)
	print("")
	
def quads2tris(F):
	out = []
	for f in F:
		if len(f) == 3: out += [f]
		elif len(f) == 4: out += [[f[0], f[1], f[2]],
								  [f[0], f[2], f[3]]]
		else: sys.exit()
	return np.array(out, np.int32)