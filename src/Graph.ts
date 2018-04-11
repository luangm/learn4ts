import Expression from "./expression/Expression";
import Session from "./Session";
import ExpressionFactory from "./expression/ExpressionFactory";

export default class Graph {

  private readonly _factory: ExpressionFactory;
  private readonly _name: string;
  private readonly _nodeMap: Map<number, Expression>;
  private readonly _paramMap: Map<string, Expression>;
  private _session: Session;

  get factory() {
    return this._factory;
  }

  get name() {
    return this._name;
  }

  get nodes() {
    return this._nodeMap;
  }

  get session() {
    return this._session;
  }

  set session(value) {
    this._session = value;
  }

  constructor(name: string) {
    this._name = name;
    this._nodeMap = new Map<number, Expression>();
    this._paramMap = new Map<string, Expression>();
    this._factory = new ExpressionFactory(this);
    this._session = new Session(this);
  }

  addNode(node: Expression): Expression {

    console.log("addNode: ", node);

    // find existing
    let existing = this.findNode(node);
    if (existing) {
      console.warn("Got Existing Node: ", node, existing);
      return existing;
    }

    // not found, add to node map and param map
    this._nodeMap.set(node.id, node);
    let paramJson = JSON.stringify(node.params);
    this._paramMap.set(paramJson, node);
    node.finalize();
    return node;
  }

  /**
   * given a node, try to find if this node (or equivalent) already exists in graph.
   */
  findNode(node: Expression): Expression | undefined {
    let params = JSON.stringify(node.params);
    return this._paramMap.get(params);
  }

  getNode(id: number): Expression | undefined {
    return this._nodeMap.get(id);
  }
}