import Graph from "../Graph";
import Visitor from "../visitor/Visitor";

export default abstract class Expression {

  static ID_COUNTER: number = 0;

  private _graph: Graph;
  private _id: number;
  private _name: string;
  private _observers: Map<number, Expression>;

  constructor(graph: Graph, name?: string) {
    this._id = Expression.ID_COUNTER++;
    this._graph = graph;
    this._name = name;
    this._observers = new Map<number, Expression>();
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

  abstract get shape(): number[];

  abstract get type(): string;

  accept(visitor: Visitor, params?: any): void {
    visitor.visit(this, params);
  }

  addObserver(observer: Expression): void {
    this._observers.set(observer.id, observer);
  }

}