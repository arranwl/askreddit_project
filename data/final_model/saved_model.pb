
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8Т
~
dense_710/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_710/kernel
w
$dense_710/kernel/Read/ReadVariableOpReadVariableOpdense_710/kernel* 
_output_shapes
:
??*
dtype0
~
dense_711/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_711/kernel
w
$dense_711/kernel/Read/ReadVariableOpReadVariableOpdense_711/kernel* 
_output_shapes
:
??*
dtype0
}
dense_712/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*!
shared_namedense_712/kernel
v
$dense_712/kernel/Read/ReadVariableOpReadVariableOpdense_712/kernel*
_output_shapes
:	?*
dtype0
|
dense_713/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_713/kernel
u
$dense_713/kernel/Read/ReadVariableOpReadVariableOpdense_713/kernel*
_output_shapes

:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_710/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_710/kernel/m
?
+Adam/dense_710/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_710/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_711/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_711/kernel/m
?
+Adam/dense_711/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_711/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_712/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdam/dense_712/kernel/m
?
+Adam/dense_712/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_712/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_713/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_713/kernel/m
?
+Adam/dense_713/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_713/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_710/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_710/kernel/v
?
+Adam/dense_710/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_710/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_711/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_711/kernel/v
?
+Adam/dense_711/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_711/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_712/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdam/dense_712/kernel/v
?
+Adam/dense_712/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_712/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_713/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_713/kernel/v
?
+Adam/dense_713/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_713/kernel/v*
_output_shapes

:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value?!B?! B?!
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
^

kernel
	variables
trainable_variables
regularization_losses
	keras_api
^

kernel
	variables
trainable_variables
regularization_losses
	keras_api
^

kernel
	variables
trainable_variables
regularization_losses
	keras_api
^

kernel
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemHmImJmKvLvMvNvO

0
1
2
3

0
1
2
3
 
?
	variables
trainable_variables
regularization_losses
$layer_metrics
%metrics
&layer_regularization_losses
'non_trainable_variables

(layers
 
\Z
VARIABLE_VALUEdense_710/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
trainable_variables
)layer_metrics
regularization_losses
*metrics
+layer_regularization_losses
,non_trainable_variables

-layers
\Z
VARIABLE_VALUEdense_711/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
trainable_variables
.layer_metrics
regularization_losses
/metrics
0layer_regularization_losses
1non_trainable_variables

2layers
\Z
VARIABLE_VALUEdense_712/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
trainable_variables
3layer_metrics
regularization_losses
4metrics
5layer_regularization_losses
6non_trainable_variables

7layers
\Z
VARIABLE_VALUEdense_713/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
trainable_variables
8layer_metrics
regularization_losses
9metrics
:layer_regularization_losses
;non_trainable_variables

<layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	?total
	@count
A	variables
B	keras_api
D
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

A	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

F	variables
}
VARIABLE_VALUEAdam/dense_710/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_711/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_712/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_713/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_710/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_711/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_712/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_713/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_input_183Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_183dense_710/kerneldense_711/kerneldense_712/kerneldense_713/kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_309679
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_710/kernel/Read/ReadVariableOp$dense_711/kernel/Read/ReadVariableOp$dense_712/kernel/Read/ReadVariableOp$dense_713/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_710/kernel/m/Read/ReadVariableOp+Adam/dense_711/kernel/m/Read/ReadVariableOp+Adam/dense_712/kernel/m/Read/ReadVariableOp+Adam/dense_713/kernel/m/Read/ReadVariableOp+Adam/dense_710/kernel/v/Read/ReadVariableOp+Adam/dense_711/kernel/v/Read/ReadVariableOp+Adam/dense_712/kernel/v/Read/ReadVariableOp+Adam/dense_713/kernel/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_309891
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_710/kerneldense_711/kerneldense_712/kerneldense_713/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_710/kernel/mAdam/dense_711/kernel/mAdam/dense_712/kernel/mAdam/dense_713/kernel/mAdam/dense_710/kernel/vAdam/dense_711/kernel/vAdam/dense_712/kernel/vAdam/dense_713/kernel/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_309964??
?
?
$__inference_signature_wrapper_309679
	input_183
unknown:
??
	unknown_0:
??
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_183unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3094722
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:??????????
#
_user_specified_name	input_183
?
?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309699

inputs<
(dense_710_matmul_readvariableop_resource:
??<
(dense_711_matmul_readvariableop_resource:
??;
(dense_712_matmul_readvariableop_resource:	?:
(dense_713_matmul_readvariableop_resource:
identity??dense_710/MatMul/ReadVariableOp?dense_711/MatMul/ReadVariableOp?dense_712/MatMul/ReadVariableOp?dense_713/MatMul/ReadVariableOp?
dense_710/MatMul/ReadVariableOpReadVariableOp(dense_710_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_710/MatMul/ReadVariableOp?
dense_710/MatMulMatMulinputs'dense_710/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_710/MatMulw
dense_710/ReluReludense_710/MatMul:product:0*
T0*(
_output_shapes
:??????????2
dense_710/Relu?
dense_711/MatMul/ReadVariableOpReadVariableOp(dense_711_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_711/MatMul/ReadVariableOp?
dense_711/MatMulMatMuldense_710/Relu:activations:0'dense_711/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_711/MatMulw
dense_711/ReluReludense_711/MatMul:product:0*
T0*(
_output_shapes
:??????????2
dense_711/Relu?
dense_712/MatMul/ReadVariableOpReadVariableOp(dense_712_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_712/MatMul/ReadVariableOp?
dense_712/MatMulMatMuldense_711/Relu:activations:0'dense_712/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_712/MatMulv
dense_712/ReluReludense_712/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_712/Relu?
dense_713/MatMul/ReadVariableOpReadVariableOp(dense_713_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_713/MatMul/ReadVariableOp?
dense_713/MatMulMatMuldense_712/Relu:activations:0'dense_713/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_713/MatMul
dense_713/SoftmaxSoftmaxdense_713/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_713/Softmaxv
IdentityIdentitydense_713/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_710/MatMul/ReadVariableOp ^dense_711/MatMul/ReadVariableOp ^dense_712/MatMul/ReadVariableOp ^dense_713/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_710/MatMul/ReadVariableOpdense_710/MatMul/ReadVariableOp2B
dense_711/MatMul/ReadVariableOpdense_711/MatMul/ReadVariableOp2B
dense_712/MatMul/ReadVariableOpdense_712/MatMul/ReadVariableOp2B
dense_713/MatMul/ReadVariableOpdense_713/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309528

inputs$
dense_710_309488:
??$
dense_711_309500:
??#
dense_712_309512:	?"
dense_713_309524:
identity??!dense_710/StatefulPartitionedCall?!dense_711/StatefulPartitionedCall?!dense_712/StatefulPartitionedCall?!dense_713/StatefulPartitionedCall?
!dense_710/StatefulPartitionedCallStatefulPartitionedCallinputsdense_710_309488*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_710_layer_call_and_return_conditional_losses_3094872#
!dense_710/StatefulPartitionedCall?
!dense_711/StatefulPartitionedCallStatefulPartitionedCall*dense_710/StatefulPartitionedCall:output:0dense_711_309500*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_711_layer_call_and_return_conditional_losses_3094992#
!dense_711/StatefulPartitionedCall?
!dense_712/StatefulPartitionedCallStatefulPartitionedCall*dense_711/StatefulPartitionedCall:output:0dense_712_309512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_712_layer_call_and_return_conditional_losses_3095112#
!dense_712/StatefulPartitionedCall?
!dense_713/StatefulPartitionedCallStatefulPartitionedCall*dense_712/StatefulPartitionedCall:output:0dense_713_309524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_713_layer_call_and_return_conditional_losses_3095232#
!dense_713/StatefulPartitionedCall?
IdentityIdentity*dense_713/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_710/StatefulPartitionedCall"^dense_711/StatefulPartitionedCall"^dense_712/StatefulPartitionedCall"^dense_713/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2F
!dense_710/StatefulPartitionedCall!dense_710/StatefulPartitionedCall2F
!dense_711/StatefulPartitionedCall!dense_711/StatefulPartitionedCall2F
!dense_712/StatefulPartitionedCall!dense_712/StatefulPartitionedCall2F
!dense_713/StatefulPartitionedCall!dense_713/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_206_layer_call_fn_309626
	input_183
unknown:
??
	unknown_0:
??
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_183unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_206_layer_call_and_return_conditional_losses_3096022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:??????????
#
_user_specified_name	input_183
?
?
!__inference__wrapped_model_309472
	input_183K
7sequential_206_dense_710_matmul_readvariableop_resource:
??K
7sequential_206_dense_711_matmul_readvariableop_resource:
??J
7sequential_206_dense_712_matmul_readvariableop_resource:	?I
7sequential_206_dense_713_matmul_readvariableop_resource:
identity??.sequential_206/dense_710/MatMul/ReadVariableOp?.sequential_206/dense_711/MatMul/ReadVariableOp?.sequential_206/dense_712/MatMul/ReadVariableOp?.sequential_206/dense_713/MatMul/ReadVariableOp?
.sequential_206/dense_710/MatMul/ReadVariableOpReadVariableOp7sequential_206_dense_710_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.sequential_206/dense_710/MatMul/ReadVariableOp?
sequential_206/dense_710/MatMulMatMul	input_1836sequential_206/dense_710/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_206/dense_710/MatMul?
sequential_206/dense_710/ReluRelu)sequential_206/dense_710/MatMul:product:0*
T0*(
_output_shapes
:??????????2
sequential_206/dense_710/Relu?
.sequential_206/dense_711/MatMul/ReadVariableOpReadVariableOp7sequential_206_dense_711_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.sequential_206/dense_711/MatMul/ReadVariableOp?
sequential_206/dense_711/MatMulMatMul+sequential_206/dense_710/Relu:activations:06sequential_206/dense_711/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
sequential_206/dense_711/MatMul?
sequential_206/dense_711/ReluRelu)sequential_206/dense_711/MatMul:product:0*
T0*(
_output_shapes
:??????????2
sequential_206/dense_711/Relu?
.sequential_206/dense_712/MatMul/ReadVariableOpReadVariableOp7sequential_206_dense_712_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype020
.sequential_206/dense_712/MatMul/ReadVariableOp?
sequential_206/dense_712/MatMulMatMul+sequential_206/dense_711/Relu:activations:06sequential_206/dense_712/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_206/dense_712/MatMul?
sequential_206/dense_712/ReluRelu)sequential_206/dense_712/MatMul:product:0*
T0*'
_output_shapes
:?????????2
sequential_206/dense_712/Relu?
.sequential_206/dense_713/MatMul/ReadVariableOpReadVariableOp7sequential_206_dense_713_matmul_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_206/dense_713/MatMul/ReadVariableOp?
sequential_206/dense_713/MatMulMatMul+sequential_206/dense_712/Relu:activations:06sequential_206/dense_713/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_206/dense_713/MatMul?
 sequential_206/dense_713/SoftmaxSoftmax)sequential_206/dense_713/MatMul:product:0*
T0*'
_output_shapes
:?????????2"
 sequential_206/dense_713/Softmax?
IdentityIdentity*sequential_206/dense_713/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp/^sequential_206/dense_710/MatMul/ReadVariableOp/^sequential_206/dense_711/MatMul/ReadVariableOp/^sequential_206/dense_712/MatMul/ReadVariableOp/^sequential_206/dense_713/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2`
.sequential_206/dense_710/MatMul/ReadVariableOp.sequential_206/dense_710/MatMul/ReadVariableOp2`
.sequential_206/dense_711/MatMul/ReadVariableOp.sequential_206/dense_711/MatMul/ReadVariableOp2`
.sequential_206/dense_712/MatMul/ReadVariableOp.sequential_206/dense_712/MatMul/ReadVariableOp2`
.sequential_206/dense_713/MatMul/ReadVariableOp.sequential_206/dense_713/MatMul/ReadVariableOp:S O
(
_output_shapes
:??????????
#
_user_specified_name	input_183
?4
?
__inference__traced_save_309891
file_prefix/
+savev2_dense_710_kernel_read_readvariableop/
+savev2_dense_711_kernel_read_readvariableop/
+savev2_dense_712_kernel_read_readvariableop/
+savev2_dense_713_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_710_kernel_m_read_readvariableop6
2savev2_adam_dense_711_kernel_m_read_readvariableop6
2savev2_adam_dense_712_kernel_m_read_readvariableop6
2savev2_adam_dense_713_kernel_m_read_readvariableop6
2savev2_adam_dense_710_kernel_v_read_readvariableop6
2savev2_adam_dense_711_kernel_v_read_readvariableop6
2savev2_adam_dense_712_kernel_v_read_readvariableop6
2savev2_adam_dense_713_kernel_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_710_kernel_read_readvariableop+savev2_dense_711_kernel_read_readvariableop+savev2_dense_712_kernel_read_readvariableop+savev2_dense_713_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_710_kernel_m_read_readvariableop2savev2_adam_dense_711_kernel_m_read_readvariableop2savev2_adam_dense_712_kernel_m_read_readvariableop2savev2_adam_dense_713_kernel_m_read_readvariableop2savev2_adam_dense_710_kernel_v_read_readvariableop2savev2_adam_dense_711_kernel_v_read_readvariableop2savev2_adam_dense_712_kernel_v_read_readvariableop2savev2_adam_dense_713_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:
??:	?:: : : : : : : : : :
??:
??:	?::
??:
??:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:$ 

_output_shapes

::&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:%!

_output_shapes
:	?:$ 

_output_shapes

::

_output_shapes
: 
?
?
*__inference_dense_711_layer_call_fn_309775

inputs
unknown:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_711_layer_call_and_return_conditional_losses_3094992
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_711_layer_call_and_return_conditional_losses_309768

inputs2
matmul_readvariableop_resource:
??
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_712_layer_call_and_return_conditional_losses_309511

inputs1
matmul_readvariableop_resource:	?
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_713_layer_call_and_return_conditional_losses_309798

inputs0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309719

inputs<
(dense_710_matmul_readvariableop_resource:
??<
(dense_711_matmul_readvariableop_resource:
??;
(dense_712_matmul_readvariableop_resource:	?:
(dense_713_matmul_readvariableop_resource:
identity??dense_710/MatMul/ReadVariableOp?dense_711/MatMul/ReadVariableOp?dense_712/MatMul/ReadVariableOp?dense_713/MatMul/ReadVariableOp?
dense_710/MatMul/ReadVariableOpReadVariableOp(dense_710_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_710/MatMul/ReadVariableOp?
dense_710/MatMulMatMulinputs'dense_710/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_710/MatMulw
dense_710/ReluReludense_710/MatMul:product:0*
T0*(
_output_shapes
:??????????2
dense_710/Relu?
dense_711/MatMul/ReadVariableOpReadVariableOp(dense_711_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_711/MatMul/ReadVariableOp?
dense_711/MatMulMatMuldense_710/Relu:activations:0'dense_711/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_711/MatMulw
dense_711/ReluReludense_711/MatMul:product:0*
T0*(
_output_shapes
:??????????2
dense_711/Relu?
dense_712/MatMul/ReadVariableOpReadVariableOp(dense_712_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02!
dense_712/MatMul/ReadVariableOp?
dense_712/MatMulMatMuldense_711/Relu:activations:0'dense_712/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_712/MatMulv
dense_712/ReluReludense_712/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_712/Relu?
dense_713/MatMul/ReadVariableOpReadVariableOp(dense_713_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_713/MatMul/ReadVariableOp?
dense_713/MatMulMatMuldense_712/Relu:activations:0'dense_713/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_713/MatMul
dense_713/SoftmaxSoftmaxdense_713/MatMul:product:0*
T0*'
_output_shapes
:?????????2
dense_713/Softmaxv
IdentityIdentitydense_713/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_710/MatMul/ReadVariableOp ^dense_711/MatMul/ReadVariableOp ^dense_712/MatMul/ReadVariableOp ^dense_713/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_710/MatMul/ReadVariableOpdense_710/MatMul/ReadVariableOp2B
dense_711/MatMul/ReadVariableOpdense_711/MatMul/ReadVariableOp2B
dense_712/MatMul/ReadVariableOpdense_712/MatMul/ReadVariableOp2B
dense_713/MatMul/ReadVariableOpdense_713/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_710_layer_call_fn_309760

inputs
unknown:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_710_layer_call_and_return_conditional_losses_3094872
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_711_layer_call_and_return_conditional_losses_309499

inputs2
matmul_readvariableop_resource:
??
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_206_layer_call_fn_309745

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_206_layer_call_and_return_conditional_losses_3096022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

*__inference_dense_712_layer_call_fn_309790

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_712_layer_call_and_return_conditional_losses_3095112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_206_layer_call_fn_309732

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_206_layer_call_and_return_conditional_losses_3095282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
"__inference__traced_restore_309964
file_prefix5
!assignvariableop_dense_710_kernel:
??7
#assignvariableop_1_dense_711_kernel:
??6
#assignvariableop_2_dense_712_kernel:	?5
#assignvariableop_3_dense_713_kernel:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: ?
+assignvariableop_13_adam_dense_710_kernel_m:
???
+assignvariableop_14_adam_dense_711_kernel_m:
??>
+assignvariableop_15_adam_dense_712_kernel_m:	?=
+assignvariableop_16_adam_dense_713_kernel_m:?
+assignvariableop_17_adam_dense_710_kernel_v:
???
+assignvariableop_18_adam_dense_711_kernel_v:
??>
+assignvariableop_19_adam_dense_712_kernel_v:	?=
+assignvariableop_20_adam_dense_713_kernel_v:
identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_710_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_711_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_712_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_713_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_dense_710_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_dense_711_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_712_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_dense_713_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_710_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_dense_711_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_712_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_dense_713_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21f
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_22?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
E__inference_dense_710_layer_call_and_return_conditional_losses_309753

inputs2
matmul_readvariableop_resource:
??
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_206_layer_call_fn_309539
	input_183
unknown:
??
	unknown_0:
??
	unknown_1:	?
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_183unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_206_layer_call_and_return_conditional_losses_3095282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
(
_output_shapes
:??????????
#
_user_specified_name	input_183
?
?
E__inference_dense_712_layer_call_and_return_conditional_losses_309783

inputs1
matmul_readvariableop_resource:	?
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulX
ReluReluMatMul:product:0*
T0*'
_output_shapes
:?????????2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309642
	input_183$
dense_710_309629:
??$
dense_711_309632:
??#
dense_712_309635:	?"
dense_713_309638:
identity??!dense_710/StatefulPartitionedCall?!dense_711/StatefulPartitionedCall?!dense_712/StatefulPartitionedCall?!dense_713/StatefulPartitionedCall?
!dense_710/StatefulPartitionedCallStatefulPartitionedCall	input_183dense_710_309629*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_710_layer_call_and_return_conditional_losses_3094872#
!dense_710/StatefulPartitionedCall?
!dense_711/StatefulPartitionedCallStatefulPartitionedCall*dense_710/StatefulPartitionedCall:output:0dense_711_309632*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_711_layer_call_and_return_conditional_losses_3094992#
!dense_711/StatefulPartitionedCall?
!dense_712/StatefulPartitionedCallStatefulPartitionedCall*dense_711/StatefulPartitionedCall:output:0dense_712_309635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_712_layer_call_and_return_conditional_losses_3095112#
!dense_712/StatefulPartitionedCall?
!dense_713/StatefulPartitionedCallStatefulPartitionedCall*dense_712/StatefulPartitionedCall:output:0dense_713_309638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_713_layer_call_and_return_conditional_losses_3095232#
!dense_713/StatefulPartitionedCall?
IdentityIdentity*dense_713/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_710/StatefulPartitionedCall"^dense_711/StatefulPartitionedCall"^dense_712/StatefulPartitionedCall"^dense_713/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2F
!dense_710/StatefulPartitionedCall!dense_710/StatefulPartitionedCall2F
!dense_711/StatefulPartitionedCall!dense_711/StatefulPartitionedCall2F
!dense_712/StatefulPartitionedCall!dense_712/StatefulPartitionedCall2F
!dense_713/StatefulPartitionedCall!dense_713/StatefulPartitionedCall:S O
(
_output_shapes
:??????????
#
_user_specified_name	input_183
?
~
*__inference_dense_713_layer_call_fn_309805

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_713_layer_call_and_return_conditional_losses_3095232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309658
	input_183$
dense_710_309645:
??$
dense_711_309648:
??#
dense_712_309651:	?"
dense_713_309654:
identity??!dense_710/StatefulPartitionedCall?!dense_711/StatefulPartitionedCall?!dense_712/StatefulPartitionedCall?!dense_713/StatefulPartitionedCall?
!dense_710/StatefulPartitionedCallStatefulPartitionedCall	input_183dense_710_309645*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_710_layer_call_and_return_conditional_losses_3094872#
!dense_710/StatefulPartitionedCall?
!dense_711/StatefulPartitionedCallStatefulPartitionedCall*dense_710/StatefulPartitionedCall:output:0dense_711_309648*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_711_layer_call_and_return_conditional_losses_3094992#
!dense_711/StatefulPartitionedCall?
!dense_712/StatefulPartitionedCallStatefulPartitionedCall*dense_711/StatefulPartitionedCall:output:0dense_712_309651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_712_layer_call_and_return_conditional_losses_3095112#
!dense_712/StatefulPartitionedCall?
!dense_713/StatefulPartitionedCallStatefulPartitionedCall*dense_712/StatefulPartitionedCall:output:0dense_713_309654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_713_layer_call_and_return_conditional_losses_3095232#
!dense_713/StatefulPartitionedCall?
IdentityIdentity*dense_713/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_710/StatefulPartitionedCall"^dense_711/StatefulPartitionedCall"^dense_712/StatefulPartitionedCall"^dense_713/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2F
!dense_710/StatefulPartitionedCall!dense_710/StatefulPartitionedCall2F
!dense_711/StatefulPartitionedCall!dense_711/StatefulPartitionedCall2F
!dense_712/StatefulPartitionedCall!dense_712/StatefulPartitionedCall2F
!dense_713/StatefulPartitionedCall!dense_713/StatefulPartitionedCall:S O
(
_output_shapes
:??????????
#
_user_specified_name	input_183
?
?
E__inference_dense_713_layer_call_and_return_conditional_losses_309523

inputs0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_dense_710_layer_call_and_return_conditional_losses_309487

inputs2
matmul_readvariableop_resource:
??
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:??????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309602

inputs$
dense_710_309589:
??$
dense_711_309592:
??#
dense_712_309595:	?"
dense_713_309598:
identity??!dense_710/StatefulPartitionedCall?!dense_711/StatefulPartitionedCall?!dense_712/StatefulPartitionedCall?!dense_713/StatefulPartitionedCall?
!dense_710/StatefulPartitionedCallStatefulPartitionedCallinputsdense_710_309589*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_710_layer_call_and_return_conditional_losses_3094872#
!dense_710/StatefulPartitionedCall?
!dense_711/StatefulPartitionedCallStatefulPartitionedCall*dense_710/StatefulPartitionedCall:output:0dense_711_309592*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_711_layer_call_and_return_conditional_losses_3094992#
!dense_711/StatefulPartitionedCall?
!dense_712/StatefulPartitionedCallStatefulPartitionedCall*dense_711/StatefulPartitionedCall:output:0dense_712_309595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_712_layer_call_and_return_conditional_losses_3095112#
!dense_712/StatefulPartitionedCall?
!dense_713/StatefulPartitionedCallStatefulPartitionedCall*dense_712/StatefulPartitionedCall:output:0dense_713_309598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_713_layer_call_and_return_conditional_losses_3095232#
!dense_713/StatefulPartitionedCall?
IdentityIdentity*dense_713/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp"^dense_710/StatefulPartitionedCall"^dense_711/StatefulPartitionedCall"^dense_712/StatefulPartitionedCall"^dense_713/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2F
!dense_710/StatefulPartitionedCall!dense_710/StatefulPartitionedCall2F
!dense_711/StatefulPartitionedCall!dense_711/StatefulPartitionedCall2F
!dense_712/StatefulPartitionedCall!dense_712/StatefulPartitionedCall2F
!dense_713/StatefulPartitionedCall!dense_713/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
	input_1833
serving_default_input_183:0??????????=
	dense_7130
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?Y
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
P_default_save_signature
*Q&call_and_return_all_conditional_losses
R__call__"
_tf_keras_sequential
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"
_tf_keras_layer
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"
_tf_keras_layer
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
*W&call_and_return_all_conditional_losses
X__call__"
_tf_keras_layer
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"
_tf_keras_layer
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemHmImJmKvLvMvNvO"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
regularization_losses
$layer_metrics
%metrics
&layer_regularization_losses
'non_trainable_variables

(layers
R__call__
P_default_save_signature
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
,
[serving_default"
signature_map
$:"
??2dense_710/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
)layer_metrics
regularization_losses
*metrics
+layer_regularization_losses
,non_trainable_variables

-layers
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_711/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
.layer_metrics
regularization_losses
/metrics
0layer_regularization_losses
1non_trainable_variables

2layers
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
#:!	?2dense_712/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
3layer_metrics
regularization_losses
4metrics
5layer_regularization_losses
6non_trainable_variables

7layers
X__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
": 2dense_713/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
8layer_metrics
regularization_losses
9metrics
:layer_regularization_losses
;non_trainable_variables

<layers
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	?total
	@count
A	variables
B	keras_api"
_tf_keras_metric
^
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
?0
@1"
trackable_list_wrapper
-
A	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
):'
??2Adam/dense_710/kernel/m
):'
??2Adam/dense_711/kernel/m
(:&	?2Adam/dense_712/kernel/m
':%2Adam/dense_713/kernel/m
):'
??2Adam/dense_710/kernel/v
):'
??2Adam/dense_711/kernel/v
(:&	?2Adam/dense_712/kernel/v
':%2Adam/dense_713/kernel/v
?B?
!__inference__wrapped_model_309472	input_183"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309699
J__inference_sequential_206_layer_call_and_return_conditional_losses_309719
J__inference_sequential_206_layer_call_and_return_conditional_losses_309642
J__inference_sequential_206_layer_call_and_return_conditional_losses_309658?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_sequential_206_layer_call_fn_309539
/__inference_sequential_206_layer_call_fn_309732
/__inference_sequential_206_layer_call_fn_309745
/__inference_sequential_206_layer_call_fn_309626?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dense_710_layer_call_and_return_conditional_losses_309753?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_710_layer_call_fn_309760?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_711_layer_call_and_return_conditional_losses_309768?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_711_layer_call_fn_309775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_712_layer_call_and_return_conditional_losses_309783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_712_layer_call_fn_309790?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_713_layer_call_and_return_conditional_losses_309798?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_713_layer_call_fn_309805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_309679	input_183"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_309472r3?0
)?&
$?!
	input_183??????????
? "5?2
0
	dense_713#? 
	dense_713??????????
E__inference_dense_710_layer_call_and_return_conditional_losses_309753]0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
*__inference_dense_710_layer_call_fn_309760P0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_711_layer_call_and_return_conditional_losses_309768]0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
*__inference_dense_711_layer_call_fn_309775P0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_712_layer_call_and_return_conditional_losses_309783\0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
*__inference_dense_712_layer_call_fn_309790O0?-
&?#
!?
inputs??????????
? "???????????
E__inference_dense_713_layer_call_and_return_conditional_losses_309798[/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
*__inference_dense_713_layer_call_fn_309805N/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_sequential_206_layer_call_and_return_conditional_losses_309642j;?8
1?.
$?!
	input_183??????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309658j;?8
1?.
$?!
	input_183??????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309699g8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_206_layer_call_and_return_conditional_losses_309719g8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_206_layer_call_fn_309539];?8
1?.
$?!
	input_183??????????
p 

 
? "???????????
/__inference_sequential_206_layer_call_fn_309626];?8
1?.
$?!
	input_183??????????
p

 
? "???????????
/__inference_sequential_206_layer_call_fn_309732Z8?5
.?+
!?
inputs??????????
p 

 
? "???????????
/__inference_sequential_206_layer_call_fn_309745Z8?5
.?+
!?
inputs??????????
p

 
? "???????????
$__inference_signature_wrapper_309679@?=
? 
6?3
1
	input_183$?!
	input_183??????????"5?2
0
	dense_713#? 
	dense_713?????????