import Graph from "../../Graph";
import Expression from "../Expression";

export default abstract class TransformExpression extends Expression {

  private _base: Expression;

  constructor(base: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
  }

  get base() {
    return this._base;
  }

  get dependencies() {
    return [this._base];
  }

  get shape() {
    return this._base.shape;
  }

}