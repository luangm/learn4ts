import Tensor from "tensor4js/dist/types/Tensor";
import Graph from "../Graph";
import Visitor from "../visitor/Visitor";

export default abstract class Expression {

  static ID_COUNTER: number = 0;

  private readonly _gradMap: Map<number, Expression>;
  private readonly _graph: Graph;
  private readonly _id: number;
  private readonly _name?: string;
  private readonly _observers: Expression[];

  get dependencies(): Expression[] {
    return [];
  }

  get factory() {
    return this._graph.factory;
  }

  get graph() {
    return this._graph;
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

  abstract get shape(): number[];

  abstract get type(): string;

  get value(): Tensor {
    let result = this.graph.session.getValue(this);
    if (!result) {
      return this.eval();
    }
    return result;
  }

  set value(val: Tensor) {
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

  add(other: Expression): Expression {
    return this.factory.add(this, other);
  }

  addObserver(observer: Expression): void {
    this._observers.push(observer);
  }

  ceil(): Expression {
    return this.factory.ceil(this);
  }

  conditional(truthy: Expression, falsy: Expression): Expression {
    return this.factory.conditional(this, truthy, falsy);
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

  elu(): Expression {
    return this.factory.elu(this);
  }

  eluGrad(): Expression {
    return this.factory.eluGrad(this);
  }

  equal(other: Expression): Expression {
    return this.factory.equal(this, other);
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

  floor(): Expression {
    return this.factory.floor(this);
  }

  floorDiv(other: Expression): Expression {
    return this.factory.floorDiv(this, other);
  }

  floorMod(other: Expression): Expression {
    return this.factory.floorMod(this, other);
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

  less(other: Expression): Expression {
    return this.factory.less(this, other);
  }

  lessEqual(other: Expression): Expression {
    return this.factory.lessEqual(this, other);
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

  reciprocal(): Expression {
    return this.factory.reciprocal(this);
  }

  reciprocalGrad(): Expression {
    return this.factory.reciprocalGrad(this);
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

  truncDiv(other: Expression): Expression {
    return this.factory.truncDiv(this, other);
  }

  truncMod(other: Expression): Expression {
    return this.factory.truncMod(this, other);
  }
}