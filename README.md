Learn4js - written in TypeScript
-

This will replace learn4js.

# List of Supported Ops

| Op                | Type          | Eval  | Grad  | Test  |
| ---               | ---           | :---: | :---: | :---: |
| Add               | Arithmetic    | Y     | Y     | Y     |
| Subtract          | Arithmetic    | Y     | Y     | Y     |
| Multiply          | Arithmetic    | Y     | Y     |       |
| Divide            | Arithmetic    | Y     | Y     |       |
| Maximum           | Arithmetic    | Y     |       |       |
| Minimum           | Arithmetic    | Y     |       |       |
| Modulo            | Arithmetic    | Y     |       |       |
| MatMul            | Arithmetic    | Y     | Y     |       |
| Constant          | Core          | Y     |       |       |
| Parameter         | Core          | Y     |       |       |
| Variable          | Core          | Y     |       |       |
| ReduceSum         | Reduction     | Y     |       |       |
| ReduceMean        | Reduction     |       |       |       |
| ReduceMax         | Reduction     |       |       |       |
| ReduceMin         | Reduction     |       |       |       |
| ReduceProd        | Reduction     |       |       |       |
| ReduceLogSumExp   | Reduction     |       |       |       |
| L1Norm            | Reduction     |       |       |       |
| L2Norm            | Reduction     |       |       |       |
| InfNorm           | Reduction     |       |       |       |
| PNorm             | Reduction     |       |       |       |
| AddN              | Special       |       |       |       |
| Assign            | Special       | Y     |       |       |
| Fill              | Special       | Y     |       |       |
| Group             | Special       | Y     |       |       |
| Repeat            | Special       | Y     |       |       |
| Reshape           | Special       | Y     | Y     | Y     |
| Tile              | Special       |       |       |       |
| Slice             | Special       |       |       |       |
| Concat            | Special       |       |       |       |
| Stack             | Special       |       |       |       |
| Absolute          | Transform     | Y     | Y     |       |
| Exponential       | Transform     | Y     | Y     |       |
| Expm1             | Transform     | Y     | Y     | Y     |
| Logarithm         | Transform     | Y     | Y     |       |
| Log1p             | Transform     | Y     | Y     | Y     |
| Negate            | Transform     | Y     | Y     |       |
| Reciprocal        | Transform     | Y     | Y     | Y     |
| ReciprocalGrad    | Transform     | Y     |       |       |
| Relu              | Transform     | Y     | Y     |       |
| Elu               | Transform     |       |       |       |
| Round             | Transform     | Y     |       |       |
| Floor             | Transform     |       |       |       |
| Ceil              | Transform     |       |       |       |
| Power             | Transform     |       |       |       |
| RSqrt             | Transform     | Y     |       |       |
| Sigmoid           | Transform     | Y     | Y     |       |
| SigmoidGrad       | Transform     | Y     |       |       |
| Sign              | Transform     | Y     |       |       |
| Softmax           | Transform     | Y     |       |       |
| SoftmaxGrad       | Transform     | Y     |       |       |
| Softplus          | Transform     |       |       |       |
| Sqrt              | Transform     | Y     | Y     |       |
| SqrtGrad          | Transform     | Y     |       |       |
| Square            | Transform     | Y     | Y     |       |
| Step              | Transform     | Y     |       |       |
| Sine              | Trigonometry  | Y     | Y     | Y     |
| Cosine            | Trigonometry  | Y     | Y     | Y     |
| Tangent           | Trigonometry  | Y     | Y     | Y     |
| Sinh              | Trigonometry  | Y     | Y     | Y     |
| Cosh              | Trigonometry  | Y     | Y     | Y     |
| Tanh              | Trigonometry  | Y     | Y     | Y     |
| TangentGrad       | Trigonometry  | Y     |       |       |
| TanhGrad          | Trigonometry  | Y     |       |       |
| Asin              | Trigonometry  |       |       |       |
| Acos              | Trigonometry  |       |       |       |
| Atan              | Trigonometry  |       |       |       |
| Asinh             | Trigonometry  |       |       |       |
| Acosh             | Trigonometry  |       |       |       |
| Atanh             | Trigonometry  |       |       |       |
| Im2Col            | CNN           |       |       |       |
| Col2Im            | CNN           |       |       |       |
| Conv2d            | CNN           |       |       |       |
| MaxPool           | CNN           |       |       |       |
| AvgPool           | CNN           |       |       |       |
| ArgMin            | Index         |       |       |       |
| ArgMax            | Index         |       |       |       |
| CumSum            |               |       |       |       |
| CumProd           |               |       |       |       |


