Learn4js - written in TypeScript
-

This will replace learn4js.

# List of Supported Ops

| Op                | Type          | Eval  | Grad  | Test  |
| ---               | ---           | :---: | :---: | :---: |
| Add               | Arithmetic    | Y     | Y     | Y     |
| Subtract          | Arithmetic    | Y     | Y     | Y     |
| Multiply          | Arithmetic    | Y     | Y     | Y     |
| Divide            | Arithmetic    | Y     | Y     | Y     |
| Maximum           | Arithmetic    | Y     | Y     | Y     |
| Minimum           | Arithmetic    | Y     | Y     | Y     |
| FloorMod          | Arithmetic    | Y     | Y     | Y     |
| FloorDiv          | Arithmetic    | Y     |       |       |
| TruncateMod       | Arithmetic    | Y     |       |       |
| TruncateDiv       | Arithmetic    | Y     |       |       |
| MatMul            | Arithmetic    | Y     | Y     | Y     |
| Power             | Arithmetic    | Y     | Y     | Y     |
| Constant          | Core          | Y     |       |       |
| Parameter         | Core          | Y     |       |       |
| Variable          | Core          | Y     |       |       |
| Zeros             | Core          | Y     |       |       |
| ReduceSum         | Reduction     | Y     | Y     | Y     |
| ReduceMean        | Reduction     | Y     | Y     | Y     |
| ReduceMax         | Reduction     | Y     | Y     | Y     |
| ReduceMin         | Reduction     | Y     | Y     | Y     |
| ReduceProd        | Reduction     | Y     |       |       |
| ReduceLogSumExp   | Reduction     | Y     | Y     | Y     |
| L1Norm            | Reduction     | Y     | Y     | Y     |
| L2Norm            | Reduction     | Y     | Y     | Y     |
| InfNorm           | Reduction     | Y     | Y     | Y     |
| PNorm             | Reduction     | Y     | Y     | Y     |
| AddN              | Special       |       |       |       |
| Assign            | Special       | Y     |       |       |
| Fill              | Special       | Y     |       |       |
| Group             | Special       | Y     |       |       |
| Repeat            | Special       | Y     |       | E     |
| Reshape           | Special       | Y     | Y     | Y     |
| Tile              | Special       | Y     |       | E     |
| Transpose         | Special       | Y     | Y     | Y     |
| Slice             | Special       |       |       |       |
| Concat            | Special       |       |       |       |
| Stack             | Special       |       |       |       |
| Absolute          | Transform     | Y     | Y     | Y     |
| Exponential       | Transform     | Y     | Y     | Y     |
| Expm1             | Transform     | Y     | Y     | Y     |
| Logarithm         | Transform     | Y     | Y     | Y     |
| Log1p             | Transform     | Y     | Y     | Y     |
| Negate            | Transform     | Y     | Y     | Y     |
| Reciprocal        | Transform     | Y     | Y     | Y     |
| ReciprocalGrad    | Transform     | Y     |       |       |
| Relu              | Transform     | Y     | Y     | Y     |
| Elu               | Transform     | Y     | Y     | Y     |
| Round             | Transform     | Y     | Y     | Y     |
| Floor             | Transform     | Y     | Y     | Y     |
| Ceil              | Transform     | Y     | Y     | Y     |
| RSqrt             | Transform     | Y     |       |       |
| Sigmoid           | Transform     | Y     | Y     | Y     |
| SigmoidGrad       | Transform     | Y     |       |       |
| Sign              | Transform     | Y     | Y     | Y     |
| Softmax           | Transform     | Y     |       |       |
| SoftmaxGrad       | Transform     | Y     |       |       |
| Softplus          | Transform     | Y     | Y     | Y     |
| Sqrt              | Transform     | Y     | Y     | Y     |
| SqrtGrad          | Transform     | Y     |       |       |
| Square            | Transform     | Y     | Y     | Y     |
| Step              | Transform     | Y     | Y     | Y     |
| Sine              | Trigonometry  | Y     | Y     | Y     |
| Cosine            | Trigonometry  | Y     | Y     | Y     |
| Tangent           | Trigonometry  | Y     | Y     | Y     |
| Sinh              | Trigonometry  | Y     | Y     | Y     |
| Cosh              | Trigonometry  | Y     | Y     | Y     |
| Tanh              | Trigonometry  | Y     | Y     | Y     |
| TangentGrad       | Trigonometry  | Y     |       |       |
| TanhGrad          | Trigonometry  | Y     |       |       |
| Asin              | Trigonometry  | Y     | Y     | Y     |
| Acos              | Trigonometry  | Y     | Y     | Y     |
| Atan              | Trigonometry  | Y     | Y     | Y     |
| Asinh             | Trigonometry  | Y     | Y     | Y     |
| Acosh             | Trigonometry  | Y     | Y     | Y     |
| Atanh             | Trigonometry  | Y     | Y     | Y     |
| Im2Col            | CNN           | Y     |       |       |
| Col2Im            | CNN           | Y     |       |       |
| Conv2d            | CNN           | Y     | Y     | Y     |
| Conv2dImageGrad   | CNN           | Y     |       |       |
| Conv2dKernelGrad  | CNN           | Y     |       |       |
| MaxPool           | CNN           |       |       |       |
| AvgPool           | CNN           |       |       |       |
| Dropout           | NN            | Y     |       |       |
| ArgMin            | Index         | Y     |       |       |
| ArgMax            | Index         | Y     |       |       |
| CumSum            |               |       |       |       |
| CumProd           |               |       |       |       |
| Equal             | Comparison    | Y     |       |       |
| NotEqual          | Comparison    | Y     |       |       |
| NotEqual          | Comparison    | Y     |       |       |
| Greater           | Comparison    | Y     |       |       |
| GreaterEqual      | Comparison    | Y     |       |       |
| Less              | Comparison    | Y     |       |       |
| LessEqual         | Comparison    | Y     |       |       |
| Conditional       | Ternary       | Y     |       |       |
| IfElse            | Control       | Y     |       |       |
| WhileLoop         | Control       | Y     |       |       |
| ForLoop           | Control       |       |       |       |
| Switch            | Control       |       |       |       |
| Erf               | Math          | Y     | Y     | Y     |
| ErfGrad           | Math          | Y     |       |       |
| Erfc              | Math          | Y     | Y     | Y     |
| ErfcGrad          | Math          | Y     |       |       |
| Gamma             | Math          | Y     |       |       |
| LGamma            | Math          | Y     |       |       |
