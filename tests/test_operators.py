import numpy as np
from bilevel_imaging_toolbox import operators

x = np.array([[1,2,3],[4,5,6],[7,8,9]])

print('Forward differences')
op = operators.make_finite_differences_operator((3,3),'fn',1)
print(op.val(x)[:,:,0])
print(op.val(x)[:,:,1])
y = op.val(x)
print(op.conj(y))

print('Backward differences')
op = operators.make_finite_differences_operator((3,3),'bn',1)
print(op.val(x)[:,:,0])
print(op.val(x)[:,:,1])
y = op.val(x)
print(op.conj(y))

print('Centered differences')
op = operators.make_finite_differences_operator((3,3),'cn',1)
print(op.val(x)[:,:,0])
print(op.val(x)[:,:,1])
y = op.val(x)
print(op.conj(y))
