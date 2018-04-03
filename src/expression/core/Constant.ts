import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Constant extends Expression {

  private readonly _value: Tensor;

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