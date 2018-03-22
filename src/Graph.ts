import Expression from "./structure/Expression";
import Session from "./Session";

export default class Graph {

  private _name: string;
  private _nodes: Map<number, Expression>;
  private _session: Session;

  constructor(name: string) {
    this._name = name;
    this._nodes = new Map<number, Expression>();
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