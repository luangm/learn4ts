import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Constant extends Expression {

  private readonly _value: Tensor;

  get notDifferentiable() {
    return true;
  }

  /**
   * Note: because constants' value should not be part of the params. The only way to distinguish is through name and id.
   * Assume if name is present, use name. Otherwise use id.
   * Which means every constant when declared is unique, unless a name is specified.
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
    return this._value.shape;
  }

  get type() {
    return ExpressionTypes.Constant;
  }

  get value() {
    return this._value;
  }

  set value(val) {
    throw new Error("Cannot set constant");
  }

  constructor(value: Tensor, graph: Graph, name?: string) {
    super(graph, name);
    this._value = value;
  }

  static evaluate(expression: Expression): Tensor {
    return expression.value;
  }

}