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
| Constant          | Core          | Y     |       |       |
| Parameter         | Core          | Y     |       |       |
| Variable          | Core          | Y     |       |       |
| ReduceSum         | Reduction     | Y     | Y     | Y     |
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
| Repeat            | Special       | Y     |       | E     |
| Reshape           | Special       | Y     | Y     | Y     |
| Tile              | Special       | Y     |       | E     |
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
| Round             | Transform     | Y     |       | E     |
| Floor             | Transform     | Y     |       | E     |
| Ceil              | Transform     | Y     |       | E     |
| Power             | Transform     |       |       |       |
| RSqrt             | Transform     | Y     |       |       |
| Sigmoid           | Transform     | Y     | Y     | Y     |
| SigmoidGrad       | Transform     | Y     |       |       |
| Sign              | Transform     | Y     |       |       |
| Softmax           | Transform     | Y     |       |       |
| SoftmaxGrad       | Transform     | Y     |       |       |
| Softplus          | Transform     | Y     | Y     | Y     |
| Sqrt              | Transform     | Y     | Y     | Y     |
| SqrtGrad          | Transform     | Y     |       |       |
| Square            | Transform     | Y     | Y     | Y     |
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
| Equal             | Comparison    | Y     |       |       |
| NotEqual          | Comparison    | Y     |       |       |
| NotEqual          | Comparison    | Y     |       |       |
| Greater           | Comparison    | Y     |       |       |
| GreaterEqual      | Comparison    | Y     |       |       |
| Less              | Comparison    | Y     |       |       |
| LessEqual         | Comparison    | Y     |       |       |
| Conditional       | Ternary       | Y     |       |       |

