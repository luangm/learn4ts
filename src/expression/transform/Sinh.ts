import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Sinh extends TransformExpression {

  get type() {
    return ExpressionTypes.Sinh;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Sinh): Tensor {
    let base = node.base.value;
    return TensorMath.sinh(base);
  }

  static gradients(node: Sinh, grad: Expression): Expression[] {
    let baseGrad = node.base.cosh().multiply(grad);
    return [baseGrad];
  }
}