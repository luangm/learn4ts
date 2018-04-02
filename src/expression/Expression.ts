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

  cos(): Expression {
    return this.factory.cos(this);
  }

  cosh(): Expression {
    return this.factory.cosh(this);
  }

  divide(other: Expression): Expression {
    return this.factory.divide(this, other);
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

  getGradient(target: Expression): Expression | undefined {
    return this._gradMap.get(target.id);
  }

  softplus(): Expression {
    return this.factory.softplus(this);
  }

  log(): Expression {
    return this.factory.log(this);
  }

  tile(repeats: number[]): Expression {
    return this.factory.tile(this, repeats);
  }

  repeat(multiple: number, dimension: number = -1) {
    return this.factory.repeat(this, multiple, dimension);
  }

  log1p(): Expression {
    return this.factory.log1p(this);
  }

  matmul(other: Expression, transposeLeft: boolean, transposeRight: boolean): Expression {
    return this.factory.matmul(this, other, transposeLeft, transposeRight);
  }

  max(other: Expression): Expression {
    return this.factory.max(this, other);
  }

  min(other: Expression): Expression {
    return this.factory.min(this, other);
  }

  mod(other: Expression): Expression {
    return this.factory.mod(this, other);
  }

  multiply(other: Expression): Expression {
    return this.factory.multiply(this, other);
  }

  negate(): Expression {
    return this.factory.negate(this);
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

  elu(): Expression {
    return this.factory.elu(this);
  }

  reshape(shape: number[]): Expression {
    return this.factory.reshape(this, shape);
  }

  floor(): Expression {
    return this.factory.floor(this);
  }

  ceil(): Expression {
    return this.factory.ceil(this);
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
}