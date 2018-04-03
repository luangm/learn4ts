import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Expm1 extends TransformExpression {

  get type() {
    return ExpressionTypes.Expm1;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Expm1;
    let base = node.base.value;
    return TensorMath.expm1(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Expm1;
    let baseGrad = node.base.exp().multiply(grad);
    return [baseGrad];
  }
}