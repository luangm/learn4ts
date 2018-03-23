import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Absolute extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Absolute;
  }

  static evaluate(node: Absolute): Tensor {
    let base = node.base.value;
    return TensorMath.abs(base);
  }

  static gradients(node: Absolute, grad: Expression): Expression[] {
    let sign = node.factory.sign(node.base);
    let result = node.factory.multiply(grad, sign);
    return [result];
  }
}