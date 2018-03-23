import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";

export default class Constant extends Expression {

  private _value: Tensor;

  constructor(value: Tensor, graph: Graph, name?: string) {
    super(graph, name);
    this._value = value;
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
    throw new Error('Cannot set constant');
  }

  static evaluate(node: Constant): Tensor {
    return node.value;
  }

}