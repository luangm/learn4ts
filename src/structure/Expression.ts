import Graph from "../Graph";
import Visitor from "../visitor/Visitor";

export default abstract class Expression {

  static ID_COUNTER: number = 0;
  private _gradMap: Map<number, Expression>;
  private _graph: Graph;
  private _id: number;
  private _name: string;
  private _observers: Expression[];

  constructor(graph: Graph, name?: string) {
    this._id = ++Expression.ID_COUNTER;
    this._graph = graph;
    this._name = name;
    this._observers = [];
  }

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

  get value() {
    let session = this.graph.session;
    if (!session.isValid(this)) {
      return session.eval(this);
    }
    return session.getValue(this);
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

  divide(other: Expression): Expression {
    return this.factory.divide(this, other);
  }

  exp(): Expression {
    return this.factory.exp(this);
  }

  expm1(): Expression {
    return this.factory.expm1(this);
  }

  getGradient(target: Expression) {
    return this._gradMap ? this._gradMap.get(target.id) : null;
  }

  log(): Expression {
    return this.factory.log(this);
  }

  log1p(): Expression {
    return this.factory.log1p(this);
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

  relu(): Expression {
    return this.factory.relu(this);
  }

  round(): Expression {
    return this.factory.round(this);
  }

  rsqrt(): Expression {
    return this.factory.rsqrt(this);
  }

  setGradient(targetId: number, grad: Expression) {
    if (!this._gradMap) {
      this._gradMap = new Map<number, Expression>();
    }
    this._gradMap.set(targetId, grad);
  }

  sign(): Expression {
    return this.factory.sign(this);
  }

  sin(): Expression {
    return this.factory.sin(this);
  }

  sqrt(): Expression {
    return this.factory.sqrt(this);
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

  tanh(): Expression {
    return this.factory.tanh(this);
  }
}