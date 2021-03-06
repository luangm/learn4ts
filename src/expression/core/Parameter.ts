import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Parameter extends Expression {

  private readonly _initialValue: Tensor;

  get initialValue() {
    return this._initialValue;
  }

  get notDifferentiable() {
    return true;
  }

  /**
   * Note: because parameters' value should not be part of the params. The only way to distinguish is through name and id.
   * Assume if name is present, use name. Otherwise use id.
   * Which means every parameter when declared is unique, unless a name is specified.
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
    return this._initialValue.shape;
  }

  get type() {
    return ExpressionTypes.Parameter;
  }

  constructor(initialValue: Tensor, graph: Graph, name?: string) {
    super(graph, name);
    this._initialValue = initialValue;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Parameter;
    return node.initialValue;
  }

}