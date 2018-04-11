import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Variable extends Expression {

  private readonly _shape: number[];

  get notDifferentiable() {
    return true;
  }

  /**
   * Note: because variables' value should not be part of the params. The only way to distinguish is through name and id.
   * Assume if name is present, use name. Otherwise use id.
   * Which means every variable when declared is unique, unless a name is specified.
   */
  get params() {
    if (this.name) {
      return {
        type: this.type,
        name: this.name
      };
    } else {
      return {
        type: this.type,
        id: this.id
      };
    }
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