export const enum ExpressionTypes {

  Add = "Add",
  Divide = "Divide",
  MatMul = "MatMul",
  Maximum = "Maximum",
  Minimum = "Minimum",
  FloorMod = "FloorMod",
  FloorDiv = "FloorDiv",
  TruncateMod = "TruncateMod",
  TruncateDiv = "TruncateDiv",
  Multiply = "Multiply",
  Subtract = "Subtract",

  Constant = "Constant",
  Parameter = "Parameter",
  Variable = "Variable",
  Zeros = "Zeros",

  ReduceSum = "ReduceSum",
  ReduceMax = "ReduceMax",
  ReduceMin = "ReduceMin",
  ReduceMean = "ReduceMean",
  ReduceProd = "ReduceProd",

  Absolute = "Absolute",
  Cosine = "Cosine",
  Expm1 = "Expm1",
  Exponential = "Exponential",
  Log1p = "Log1p",
  Logarithm = "Logarithm",
  Negate = "Negate",
  Reciprocal = "Reciprocal",
  ReciprocalGrad = "ReciprocalGrad",
  Elu = "Elu",
  EluGrad = "EluGrad",
  Relu = "Relu",
  Round = "Round",
  Floor = "Floor",
  Ceil = "Ceil",
  RSqrt = "RSqrt",
  Sigmoid = "Sigmoid",
  SigmoidGrad = "SigmoidGrad",
  Sign = "Sign",
  Sine = "Sine",
  Softplus = "Softplus",
  Softmax = "Softmax",
  SoftmaxGrad = "SoftmaxGrad",
  Square = "Square",
  Sqrt = "Sqrt",
  SqrtGrad = "SqrtGrad",
  Step = "Step",
  Tangent = "Tangent",
  TangentGrad = "TangentGrad",
  Tanh = "Tanh",
  TanhGrad = "TanhGrad",
  Cosh = "Cosh",
  Sinh = "Sinh",

  Asin = "Asin",
  Asinh = "Asinh",
  Acos = "Acos",
  Acosh = "Acosh",
  Atan = "Atan",
  Atanh = "Atanh",

  Fill = "Fill",
  Assign = "Assign",
  Group = "Group",
  AddN = "AddN",
  Reshape = "Reshape",
  Repeat = "Repeat",
  Tile = "Tile",
  Slice = "Slice",

  Conditional = "Conditional",
  Equal = "Equal",
  NotEqual = "NotEqual",
  Greater = "Greater",
  GreaterEqual = "GreaterEqual",
  Less = "Less",
  LessEqual = "LessEqual",

  ArgMin = "ArgMin",
  ArgMax = "ArgMax"
}