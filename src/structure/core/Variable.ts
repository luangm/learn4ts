import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";

export default class Variable extends Expression {

  private _shape: number[];

  constructor(shape: number[], graph: Graph, name?: string) {
    super(graph, name);
    this._shape = shape;
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Variable;
  }

}