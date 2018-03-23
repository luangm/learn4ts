import Graph from "../Graph";
import Visitor from "../visitor/Visitor";

export default abstract class Expression {

  static ID_COUNTER: number = 0;

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

  accept(visitor: Visitor, params?: any): void {
    visitor.visit(this, params);
  }

  addObserver(observer: Expression): void {
    this._observers.push(observer);
  }

}