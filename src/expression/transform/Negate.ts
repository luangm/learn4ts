import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Negate extends TransformExpression {

  get type() {
    return ExpressionTypes.Negate;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Negate;
    let base = node.base.value;
    return TensorMath.negate(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Negate;
    let baseGrad = grad.negate();
    return [baseGrad];
  }
}