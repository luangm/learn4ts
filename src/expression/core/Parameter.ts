import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";

export default class Parameter extends Expression {

  private _initialValue: Tensor;

  constructor(initialValue: Tensor, graph: Graph, name?: string) {
    super(graph, name);
    this._initialValue = initialValue;
  }

  get initialValue() {
    return this._initialValue;
  }

  get shape() {
    return this._initialValue.shape;
  }

  get type() {
    return ExpressionTypes.Parameter;
  }

  static evaluate(node: Parameter): Tensor {
    return node.initialValue;
  }

}