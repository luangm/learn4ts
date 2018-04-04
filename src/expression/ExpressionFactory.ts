import {ShapeUtils, Tensor} from "tensor4js";
import Graph from "../Graph";
import Add from "./binary/Add";
import Divide from "./binary/Divide";
import MatMul from "./binary/MatMul";
import Maximum from "./binary/Maximum";
import Minimum from "./binary/Minimum";
import FloorMod from "./binary/FloorMod";
import Multiply from "./binary/Multiply";
import Subtract from "./binary/Subtract";
import Constant from "./core/Constant";
import Parameter from "./core/Parameter";
import Variable from "./core/Variable";
import Expression from "./Expression";
import ReduceSum from "./reduction/ReduceSum";
import AddN from "./special/AddN";
import Assign from "./special/Assign";
import Fill from "./special/Fill";
import Group from "./special/Group";
import Reshape from "./special/Reshape";
import Absolute from "./transform/Absolute";
import Cosh from "./transform/Cosh";
import Cosine from "./transform/Cosine";
import Expm1 from "./transform/Expm1";
import Exponential from "./transform/Exponential";
import Log1p from "./transform/Log1p";
import Logarithm from "./transform/Logarithm";
import Negate from "./transform/Negate";
import Reciprocal from "./transform/Reciprocal";
import ReciprocalGrad from "./transform/ReciprocalGrad";
import Relu from "./transform/Relu";
import Round from "./transform/Round";
import RSqrt from "./transform/RSqrt";
import Sigmoid from "./transform/Sigmoid";
import SigmoidGrad from "./transform/SigmoidGrad";
import Sign from "./transform/Sign";
import Sine from "./transform/Sine";
import Sinh from "./transform/Sinh";
import Softmax from "./transform/Softmax";
import SoftmaxGrad from "./transform/SoftmaxGrad";
import Sqrt from "./transform/Sqrt";
import SqrtGrad from "./transform/SqrtGrad";
import Square from "./transform/Square";
import Step from "./transform/Step";
import Tangent from "./transform/Tangent";
import TangentGrad from "./transform/TangentGrad";
import Tanh from "./transform/Tanh";
import TanhGrad from "./transform/TanhGrad";
import Elu from "./transform/Elu";
import Floor from "./transform/Floor";
import Ceil from "./transform/Ceil";
import Softplus from "./transform/Softplus";
import Repeat from "./special/Repeat";
import Tile from "./special/Tile";
import EluGrad from "./transform/EluGrad";
import Conditional from "./special/Conditional";
import Greater from "./comparison/Greater";
import GreaterEqual from "./comparison/GreaterEqual";
import Less from "./comparison/Less";
import LessEqual from "./comparison/LessEqual";
import Equal from "./comparison/Equal";
import NotEqual from "./comparison/NotEqual";
import FloorDiv from "./binary/FloorDiv";
import TruncateMod from "./binary/TruncateMod";
import TruncateDiv from "./binary/TruncateDiv";
import Zeros from "./core/Zeros";
import Asin from "./transform/Asin";
import Asinh from "./transform/Asinh";
import Acos from "./transform/Acos";
import Acosh from "./transform/Acosh";
import Atan from "./transform/Atan";
import Atanh from "./transform/Atanh";
import ReduceMean from "./reduction/ReduceMean";
import ReduceMax from "./reduction/ReduceMax";
import ReduceMin from "./reduction/ReduceMin";
import ReduceProd from "./reduction/ReduceProd";

export default class ExpressionFactory {

  private readonly _graph: Graph;

  get graph() {
    return this._graph;
  }

  constructor(graph: Graph) {
    this._graph = graph;
  }

  abs(base: Expression, name?: string): Expression {
    return this.addNode(new Absolute(base, this.graph, name), base);
  }

  acos(base: Expression, name?: string): Expression {
    return this.addNode(new Acos(base, this.graph, name), base);
  }

  acosh(base: Expression, name?: string): Expression {
    return this.addNode(new Acosh(base, this.graph, name), base);
  }

  add(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Add(left, right, this.graph, name), left, right);
  }

  addN(list: Expression[], name?: string): Expression {
    if (list.length === 1) {
      return list[0];
    }
    return this.addNode(new AddN(list, this.graph, name), ...list);
  }

  asin(base: Expression, name?: string): Expression {
    return this.addNode(new Asin(base, this.graph, name), base);
  }

  asinh(base: Expression, name?: string): Expression {
    return this.addNode(new Asinh(base, this.graph, name), base);
  }

  assign(ref: Expression, source: Expression, name?: string): Expression {
    return this.addNode(new Assign(ref, source, this.graph, name), source);
  }

  atan(base: Expression, name?: string): Expression {
    return this.addNode(new Atan(base, this.graph, name), base);
  }

  atanh(base: Expression, name?: string): Expression {
    return this.addNode(new Atanh(base, this.graph, name), base);
  }

  ceil(base: Expression, name?: string): Expression {
    return this.addNode(new Ceil(base, this.graph, name), base);
  }

  conditional(condition: Expression, truthy: Expression, falsy: Expression, name?: string): Expression {
    return this.addNode(new Conditional(condition, truthy, falsy, this.graph, name), condition, truthy, falsy);
  }

  constant(value: Tensor, name?: string): Expression {
    return this.addNode(new Constant(value, this.graph, name));
  }

  cos(base: Expression, name?: string): Expression {
    return this.addNode(new Cosine(base, this.graph, name), base);
  }

  cosh(base: Expression, name?: string): Expression {
    return this.addNode(new Cosh(base, this.graph, name), base);
  }

  divide(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Divide(left, right, this.graph, name), left, right);
  }

  elu(base: Expression, name?: string): Expression {
    return this.addNode(new Elu(base, this.graph, name), base);
  }

  eluGrad(base: Expression, name?: string): Expression {
    return this.addNode(new EluGrad(base, this.graph, name), base);
  }

  equal(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Equal(left, right, this.graph, name), left, right);
  }

  exp(base: Expression, name?: string): Expression {
    return this.addNode(new Exponential(base, this.graph, name), base);
  }

  expm1(base: Expression, name?: string): Expression {
    return this.addNode(new Expm1(base, this.graph, name), base);
  }

  fill(scalar: number, shape: number[], name?: string): Expression {
    return this.addNode(new Fill(scalar, shape, this.graph, name));
  }

  floor(base: Expression, name?: string): Expression {
    return this.addNode(new Floor(base, this.graph, name), base);
  }

  floorDiv(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new FloorDiv(left, right, this.graph, name), left, right);
  }

  floorMod(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new FloorMod(left, right, this.graph, name), left, right);
  }

  greater(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Greater(left, right, this.graph, name), left, right);
  }

  greaterEqual(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new GreaterEqual(left, right, this.graph, name), left, right);
  }

  group(list: Expression[], name?: string): Expression {
    return this.addNode(new Group(list, this.graph, name), ...list);
  }

  less(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Less(left, right, this.graph, name), left, right);
  }

  lessEqual(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new LessEqual(left, right, this.graph, name), left, right);
  }

  log(base: Expression, name?: string): Expression {
    return this.addNode(new Logarithm(base, this.graph, name), base);
  }

  log1p(base: Expression, name?: string): Expression {
    return this.addNode(new Log1p(base, this.graph, name), base);
  }

  matmul(left: Expression, right: Expression, transposeLeft: boolean, transposeRight: boolean, name?: string): Expression {
    return this.addNode(new MatMul(left, right, transposeLeft, transposeRight, this.graph, name), left, right);
  }

  max(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Maximum(left, right, this.graph, name), left, right);
  }

  min(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Minimum(left, right, this.graph, name), left, right);
  }

  multiply(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Multiply(left, right, this.graph, name), left, right);
  }

  negate(base: Expression, name?: string): Expression {
    return this.addNode(new Negate(base, this.graph, name), base);
  }

  notEqual(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new NotEqual(left, right, this.graph, name), left, right);
  }

  parameter(value: Tensor, name?: string): Expression {
    return this.addNode(new Parameter(value, this.graph, name));
  }

  reciprocal(base: Expression, name?: string): Expression {
    return this.addNode(new Reciprocal(base, this.graph, name), base);
  }

  reciprocalGrad(base: Expression, name?: string): Expression {
    return this.addNode(new ReciprocalGrad(base, this.graph, name), base);
  }

  reduceMax(base: Expression, dims: number | number[], name?: string) {
    return this.addNode(new ReduceMax(base, dims, this.graph, name), base);
  }

  reduceMean(base: Expression, dims: number | number[], name?: string) {
    return this.addNode(new ReduceMean(base, dims, this.graph, name), base);
  }

  reduceMin(base: Expression, dims: number | number[], name?: string) {
    return this.addNode(new ReduceMin(base, dims, this.graph, name), base);
  }

  reduceProd(base: Expression, dims: number | number[], name?: string) {
    return this.addNode(new ReduceProd(base, dims, this.graph, name), base);
  }

  reduceSum(base: Expression, dims: number | number[], name?: string) {
    return this.addNode(new ReduceSum(base, dims, this.graph, name), base);
  }

  relu(base: Expression, name?: string): Expression {
    return this.addNode(new Relu(base, this.graph, name), base);
  }

  repeat(base: Expression, multiple: number, dimension: number = -1, name?: string): Expression {
    return this.addNode(new Repeat(base, multiple, dimension, this.graph, name), base);
  }

  reshape(base: Expression, shape: number[], name?: string): Expression {
    if (ShapeUtils.shapeEquals(base.shape, shape)) {
      return base;
    }
    return this.addNode(new Reshape(base, shape, this.graph, name), base);
  }

  round(base: Expression, name?: string): Expression {
    return this.addNode(new Round(base, this.graph, name), base);
  }

  rsqrt(base: Expression, name?: string): Expression {
    return this.addNode(new RSqrt(base, this.graph, name), base);
  }

  sigmoid(base: Expression, name?: string): Expression {
    return this.addNode(new Sigmoid(base, this.graph, name), base);
  }

  sigmoidGrad(base: Expression, name?: string): Expression {
    return this.addNode(new SigmoidGrad(base, this.graph, name), base);
  }

  sign(base: Expression, name?: string): Expression {
    return this.addNode(new Sign(base, this.graph, name), base);
  }

  sin(base: Expression, name?: string): Expression {
    return this.addNode(new Sine(base, this.graph, name), base);
  }

  sinh(base: Expression, name?: string): Expression {
    return this.addNode(new Sinh(base, this.graph, name), base);
  }

  softmax(base: Expression, name?: string): Expression {
    return this.addNode(new Softmax(base, this.graph, name), base);
  }

  softmaxGrad(base: Expression, name?: string): Expression {
    return this.addNode(new SoftmaxGrad(base, this.graph, name), base);
  }

  softplus(base: Expression, name?: string): Expression {
    return this.addNode(new Softplus(base, this.graph, name), base);
  }

  sqrt(base: Expression, name?: string): Expression {
    return this.addNode(new Sqrt(base, this.graph, name), base);
  }

  sqrtGrad(base: Expression, name?: string): Expression {
    return this.addNode(new SqrtGrad(base, this.graph, name), base);
  }

  square(base: Expression, name?: string): Expression {
    return this.addNode(new Square(base, this.graph, name), base);
  }

  step(base: Expression, name?: string): Expression {
    return this.addNode(new Step(base, this.graph, name), base);
  }

  subtract(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Subtract(left, right, this.graph, name), left, right);
  }

  tan(base: Expression, name?: string): Expression {
    return this.addNode(new Tangent(base, this.graph, name), base);
  }

  tanGrad(base: Expression, name?: string): Expression {
    return this.addNode(new TangentGrad(base, this.graph, name), base);
  }

  tanh(base: Expression, name?: string): Expression {
    return this.addNode(new Tanh(base, this.graph, name), base);
  }

  tanhGrad(base: Expression, name?: string): Expression {
    return this.addNode(new TanhGrad(base, this.graph, name), base);
  }

  tile(base: Expression, repeats: number[], name?: string): Expression {
    return this.addNode(new Tile(base, repeats, this.graph, name), base);
  }

  truncDiv(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new TruncateDiv(left, right, this.graph, name), left, right);
  }

  truncMod(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new TruncateMod(left, right, this.graph, name), left, right);
  }

  variable(shape: number[], name?: string): Expression {
    return this.addNode(new Variable(shape, this.graph, name));
  }

  zeros(shape: number[], name?: string) {
    return this.addNode(new Zeros(shape, this.graph, name));
  }

  private addNode(node: Expression, ...dependencies: Expression[]): Expression {
    let result = this.graph.addNode(node);
    for (let dep of dependencies) {
      dep.addObserver(result);
    }
    return result;
  }
}