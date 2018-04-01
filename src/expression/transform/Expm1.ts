import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Expm1 extends TransformExpression {

  get type() {
    return ExpressionTypes.Expm1;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Expm1): Tensor {
    let base = node.base.value;
    return TensorMath.expm1(base);
  }

  static gradients(node: Expm1, grad: Expression): Expression[] {
    let baseGrad = node.base.exp().multiply(grad);
    return [baseGrad];
  }
}