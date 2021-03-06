import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";
import Absolute from "./Absolute";

export default class Floor extends TransformExpression {

  get type() {
    return ExpressionTypes.Floor;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Floor;
    let base = node.base.value;
    return TensorMath.floor(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    return [expression.zeros()];
  }
}