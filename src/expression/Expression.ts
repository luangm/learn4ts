import Tensor from "tensor4js/dist/types/Tensor";
import Graph from "../Graph";
import Visitor from "../visitor/Visitor";
import {ShapeUtils} from "tensor4js";
import {Conv2dOptions} from "./nn/Conv2d";
import {Im2ColOptions} from "./nn/Im2Col";
import {Col2ImOptions} from "./nn/Col2Im";

export default abstract class Expression {

  static ID_COUNTER: number = 0;

  private readonly _gradMap: Map<number, Expression>;
  private readonly _graph: Graph;
  private readonly _id: number;
  private readonly _name?: string;
  private readonly _observers: Expression[];
  private _subExpression: Expression | undefined;

  get dependencies(): Expression[] {
    return [];
  }

  get factory() {
    return this._graph.factory;
  }

  get graph() {
    return this._graph;
  }

  get hasGradients(): boolean {
    return this._gradMap.size > 0;
  }

  get id() {
    return this._id;
  }

  get name() {
    return this._name;
  }

  get observers(): Expression[] {
    return this._observers;
  }

  /**
   * The .params property SHOULD return a ONE-LEVEL object.
   * The property will be turned into JSON string at the graph for comparison.
   */
  get params(): object {
    return {
      type: this.type,
      name: this.name
    };
  }

  abstract get shape(): number[];

  get subExpression(): Expression | undefined {
    return this._subExpression;
  }

  abstract get type(): string;

  get value(): Tensor {
    let result = this.graph.session.getValue(this);
    if (!result) {
      return this.eval();
    }
    return result;
  }

  set value(val: Tensor) {
    if (!ShapeUtils.shapeEquals(val.shape, this.shape)) {
      throw new Error("Cannot assign value with a different shape.");
    }
    this.graph.session.setValue(this, val);
  }

  protected constructor(graph: Graph, name?: string) {
    this._id = ++Expression.ID_COUNTER;
    this._graph = graph;
    this._name = name;
    this._observers = [];
    this._gradMap = new Map<number, Expression>();
  }

  abs(): Expression {
    return this.factory.abs(this);
  }

  accept(visitor: Visitor, params?: any): void {
    visitor.visit(this, params);
  }

  acos(): Expression {
    return this.factory.acos(this);
  }

  acosh(): Expression {
    return this.factory.acosh(this);
  }

  add(other: Expression): Expression {
    return this.factory.add(this, other);
  }

  addObserver(observer: Expression): void {
    this._observers.push(observer);
  }

  asin(): Expression {
    return this.factory.asin(this);
  }

  asinh(): Expression {
    return this.factory.asinh(this);
  }

  assign(newValue: Expression): Expression {
    return this.factory.assign(this, newValue);
  }

  atan(): Expression {
    return this.factory.atan(this);
  }

  atanh(): Expression {
    return this.factory.atanh(this);
  }

  ceil(): Expression {
    return this.factory.ceil(this);
  }

  col2im(options: Col2ImOptions) {
    return this.factory.col2im(this, options);
  }

  conditional(truthy: Expression, falsy: Expression): Expression {
    return this.factory.conditional(this, truthy, falsy);
  }

  conv2d(kernel: Expression, options: Conv2dOptions): Expression {
    return this.factory.conv2d(this, kernel, options);
  }

  cos(): Expression {
    return this.factory.cos(this);
  }

  cosh(): Expression {
    return this.factory.cosh(this);
  }

  divide(other: Expression): Expression {
    return this.factory.divide(this, other);
  }

  dropout(probability: number = 0.5): Expression {
    return this.factory.dropout(this, probability);
  }

  elu(): Expression {
    return this.factory.elu(this);
  }

  eluGrad(): Expression {
    return this.factory.eluGrad(this);
  }

  equal(other: Expression): Expression {
    return this.factory.equal(this, other);
  }

  erf(): Expression {
    return this.factory.erf(this);
  }

  erfGrad(): Expression {
    return this.factory.erfGrad(this);
  }

  erfc(): Expression {
    return this.factory.erfc(this);
  }

  erfcGrad(): Expression {
    return this.factory.erfcGrad(this);
  }

  eval(): Tensor {
    return this.graph.session.eval(this);
  }

  exp(): Expression {
    return this.factory.exp(this);
  }

  expm1(): Expression {
    return this.factory.expm1(this);
  }

  finalize() {
    let sub = this.buildSubExpression();
    if (sub) {
      this._subExpression = sub;
    }
  }

  floor(): Expression {
    return this.factory.floor(this);
  }

  floorDiv(other: Expression): Expression {
    return this.factory.floorDiv(this, other);
  }

  floorMod(other: Expression): Expression {
    return this.factory.floorMod(this, other);
  }

  gamma(): Expression {
    return this.factory.gamma(this);
  }

  getGradient(target: Expression): Expression | undefined {
    return this._gradMap.get(target.id);
  }

  greater(other: Expression): Expression {
    return this.factory.greater(this, other);
  }

  greaterEqual(other: Expression): Expression {
    return this.factory.greaterEqual(this, other);
  }

  im2col(options: Im2ColOptions) {
    return this.factory.im2col(this, options);
  }

  less(other: Expression): Expression {
    return this.factory.less(this, other);
  }

  lessEqual(other: Expression): Expression {
    return this.factory.lessEqual(this, other);
  }

  lgamma(): Expression {
    return this.factory.lgamma(this);
  }

  log(): Expression {
    return this.factory.log(this);
  }

  log1p(): Expression {
    return this.factory.log1p(this);
  }

  matmul(other: Expression, transposeLeft: boolean = false, transposeRight: boolean = false): Expression {
    return this.factory.matmul(this, other, transposeLeft, transposeRight);
  }

  max(other: Expression): Expression {
    return this.factory.max(this, other);
  }

  min(other: Expression): Expression {
    return this.factory.min(this, other);
  }

  multiply(other: Expression): Expression {
    return this.factory.multiply(this, other);
  }

  negate(): Expression {
    return this.factory.negate(this);
  }

  notEqual(other: Expression): Expression {
    return this.factory.notEqual(this, other);
  }

  pow(other: Expression): Expression {
    return this.factory.pow(this, other);
  }

  reciprocal(): Expression {
    return this.factory.reciprocal(this);
  }

  reciprocalGrad(): Expression {
    return this.factory.reciprocalGrad(this);
  }

  reduceLogSumExp(dims: number | number[] = -1): Expression {
    return this.factory.reduceLogSumExp(this, dims);
  }

  reduceMax(dims: number | number[] = -1): Expression {
    return this.factory.reduceMax(this, dims);
  }

  reduceMean(dims: number | number[] = -1): Expression {
    return this.factory.reduceMean(this, dims);
  }

  reduceMin(dims: number | number[] = -1): Expression {
    return this.factory.reduceMin(this, dims);
  }

  reduceProd(dims: number | number[] = -1): Expression {
    return this.factory.reduceProd(this, dims);
  }

  reduceSum(dims: number | number[] = -1): Expression {
    return this.factory.reduceSum(this, dims);
  }

  relu(): Expression {
    return this.factory.relu(this);
  }

  repeat(multiple: number, dimension: number = -1) {
    return this.factory.repeat(this, multiple, dimension);
  }

  reshape(shape: number[]): Expression {
    return this.factory.reshape(this, shape);
  }

  round(): Expression {
    return this.factory.round(this);
  }

  rsqrt(): Expression {
    return this.factory.rsqrt(this);
  }

  setGradient(targetId: number, grad: Expression) {
    this._gradMap.set(targetId, grad);
  }

  sigmoid(): Expression {
    return this.factory.sigmoid(this);
  }

  sigmoidGrad(): Expression {
    return this.factory.sigmoidGrad(this);
  }

  sign(): Expression {
    return this.factory.sign(this);
  }

  sin(): Expression {
    return this.factory.sin(this);
  }

  sinh(): Expression {
    return this.factory.sinh(this);
  }

  softplus(): Expression {
    return this.factory.softplus(this);
  }

  sqrt(): Expression {
    return this.factory.sqrt(this);
  }

  sqrtGrad(): Expression {
    return this.factory.sqrtGrad(this);
  }

  square(): Expression {
    return this.factory.square(this);
  }

  step(): Expression {
    return this.factory.step(this);
  }

  subtract(other: Expression): Expression {
    return this.factory.subtract(this, other);
  }

  tan(): Expression {
    return this.factory.tan(this);
  }

  tanGrad(): Expression {
    return this.factory.tanGrad(this);
  }

  tanh(): Expression {
    return this.factory.tanh(this);
  }

  tanhGrad(): Expression {
    return this.factory.tanhGrad(this);
  }

  tile(repeats: number[]): Expression {
    return this.factory.tile(this, repeats);
  }

  transpose(newAxis: number[] = []): Expression {
    return this.factory.transpose(this, newAxis);
  }

  truncDiv(other: Expression): Expression {
    return this.factory.truncDiv(this, other);
  }

  truncMod(other: Expression): Expression {
    return this.factory.truncMod(this, other);
  }

  zeros(): Expression {
    return this.factory.zeros(this.shape);
  }

  /**
   * If child class override this and return an Expression,
   * Then the subExpression is assumed to be equivalent to this.
   * Evaluation visitor will use the subExpression
   */
  protected buildSubExpression(): Expression | undefined {
    return undefined;
  }
}