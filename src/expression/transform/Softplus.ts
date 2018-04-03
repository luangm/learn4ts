import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Softplus extends TransformExpression {

  get type() {
    return ExpressionTypes.Softplus;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Softplus;
    let base = node.base.value;
    return TensorMath.softplus(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Softplus;
    let baseGrad = node.base.sigmoid().multiply(grad);
    return [baseGrad];
  }
}