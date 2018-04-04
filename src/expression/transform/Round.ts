import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Round extends TransformExpression {

  get type() {
    return ExpressionTypes.Round;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Round;
    let base = node.base.value;
    return TensorMath.round(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    return [expression.zeros()];
  }
}