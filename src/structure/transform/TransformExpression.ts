import Expression from "../Expression";
import Graph from "../../Graph";

export default abstract class TransformExpression extends Expression {

  private _base: Expression;

  constructor(base: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
  }

  get base() {
    return this._base;
  }

  get shape() {
    return this.base.shape;
  }

}