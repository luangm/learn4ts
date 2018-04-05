import Graph from "../../Graph";
import Expression from "../Expression";

export default abstract class TransformExpression extends Expression {

  private readonly _base: Expression;

  get base() {
    return this._base;
  }

  get dependencies() {
    return [this._base];
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      base: this.base.id
    };
  }

  get shape() {
    return this._base.shape;
  }

  protected constructor(base: Expression, graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
  }

}