import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Absolute extends TransformExpression {

  get type() {
    return ExpressionTypes.Absolute;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Absolute;
    let base = node.base.value;
    return TensorMath.abs(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Absolute;
    let baseGrad = node.base.sign().multiply(grad);
    return [baseGrad];
  }
}