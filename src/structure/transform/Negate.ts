import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Negate extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Negate;
  }

  static evaluate(node: Negate): Tensor {
    let base = node.base.value;
    return TensorMath.negate(base);
  }

  static gradients(node: Negate, grad: Expression): Expression[] {
    let baseGrad = grad.negate();
    return [baseGrad];
  }
}