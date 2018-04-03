import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Parameter extends Expression {

  private readonly _initialValue: Tensor;

  get initialValue() {
    return this._initialValue;
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