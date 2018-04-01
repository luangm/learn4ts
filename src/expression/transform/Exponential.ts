import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Exponential extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Exponential;
  }

  static evaluate(node: Exponential): Tensor {
    let base = node.base.value;
    return TensorMath.exp(base);
  }

  static gradients(node: Exponential, grad: Expression): Expression[] {
    let baseGrad = node.multiply(grad);
    return [baseGrad];
  }
}