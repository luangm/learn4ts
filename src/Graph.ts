import Expression from "./expression/Expression";
import Session from "./Session";
import ExpressionFactory from "./expression/ExpressionFactory";

export default class Graph {

  private _factory: ExpressionFactory;
  private _name: string;
  private _nodes: Map<number, Expression>;
  private _session: Session;

  constructor(name: string) {
    this._name = name;
    this._nodes = new Map<number, Expression>();
    this._factory = new ExpressionFactory(this);
  }

  get factory() {
    return this._factory;
  }

  get name() {
    return this._name;
  }

  get session() {
    return this._session;
  }

  set session(value) {
    this._session = value;
  }

  addNode(node: Expression): Expression {
    this._nodes.set(node.id, node);
    return node;
  }

  getNode(id: number) {
    return this._nodes.get(id);
  }
}