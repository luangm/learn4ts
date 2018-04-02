import {ShapeUtils, Tensor} from "tensor4js";
import Graph from "../Graph";
import Add from "./binary/Add";
import Divide from "./binary/Divide";
import MatMul from "./binary/MatMul";
import Maximum from "./binary/Maximum";
import Minimum from "./binary/Minimum";
import Modulo from "./binary/Modulo";
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

  add(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Add(left, right, this.graph, name), left, right);
  }

  addN(list: Expression[], name?: string): Expression {
    if (list.length === 1) {
      return list[0];
    }
    return this.addNode(new AddN(list, this.graph, name), ...list);
  }

  assign(ref: Expression, source: Expression, name?: string): Expression {
    return this.addNode(new Assign(ref, source, this.graph, name), source);
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

  exp(base: Expression, name?: string): Expression {
    return this.addNode(new Exponential(base, this.graph, name), base);
  }

  expm1(base: Expression, name?: string): Expression {
    return this.addNode(new Expm1(base, this.graph, name), base);
  }

  fill(scalar: number, shape: number[], name?: string): Expression {
    return this.addNode(new Fill(scalar, shape, this.graph, name));
  }

  group(list: Expression[], name?: string): Expression {
    return this.addNode(new Group(list, this.graph, name), ...list);
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

  mod(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Modulo(left, right, this.graph, name), left, right);
  }

  multiply(left: Expression, right: Expression, name?: string): Expression {
    return this.addNode(new Multiply(left, right, this.graph, name), left, right);
  }

  negate(base: Expression, name?: string): Expression {
    return this.addNode(new Negate(base, this.graph, name), base);
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

  reduceSum(base: Expression, dims: number | number[], name?: string) {
    return dims == null ? base : this.addNode(new ReduceSum(base, dims, this.graph, name), base);
  }

  relu(base: Expression, name?: string): Expression {
    return this.addNode(new Relu(base, this.graph, name), base);
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

  variable(shape: number[], name?: string): Expression {
    return this.addNode(new Variable(shape, this.graph, name));
  }

  private addNode(node: Expression, ...dependencies: Expression[]): Expression {
    let result = this.graph.addNode(node);
    for (let dep of dependencies) {
      dep.addObserver(result);
    }
    return result;
  }
}