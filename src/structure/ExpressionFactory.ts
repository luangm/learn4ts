import Graph from "../Graph";
import Expression from "./Expression";
import Add from "./binary/Add";
import Absolute from "./transform/Absolute";

export default class ExpressionFactory {

  private _graph: Graph;

  constructor(graph: Graph) {
    this._graph = graph;
  }

  get graph() {
    return this._graph;
  }

  abs(base: Expression, name?: string) {
    return this.addNode(new Absolute(base, this.graph, name), base);
  }

  add(left: Expression, right: Expression, name?: string) {
    return this.addNode(new Add(left, right, this.graph, name), left, right);
  }

  private addNode(node: Expression, ...dependencies: Expression[]) {
    let result = this.graph.addNode(node);
    for (let dep of dependencies) {
      dep.addObserver(result);
    }
    return result;
  }
}