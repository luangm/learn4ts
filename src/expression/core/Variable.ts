import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Variable extends Expression {

  private readonly _shape: number[];

  get params() {
    return {
      type: this.type,
      name: this.name,
      shape: this.shape
    };
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Variable;
  }

  constructor(shape: number[], graph: Graph, name?: string) {
    super(graph, name);
    this._shape = shape;
  }

}