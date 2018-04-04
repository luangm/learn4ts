import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Zeros extends Expression {

  private readonly _value: Tensor;

  get shape() {
    return this._value.shape;
  }

  get type() {
    return ExpressionTypes.Zeros;
  }

  get value() {
    return this._value;
  }

  set value(val) {
    throw new Error("Cannot set value to zeros");
  }

  constructor(shape: number[], graph: Graph, name?: string) {
    super(graph, name);
    this._value = Tensor.sparseZeros(shape);
  }

  static evaluate(expression: Expression): Tensor {
    return expression.value;
  }

}