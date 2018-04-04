import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Acosh extends TransformExpression {

  get type() {
    return ExpressionTypes.Acosh;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Acosh;
    let base = node.base.value;
    return TensorMath.acosh(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Acosh;
    let one = node.factory.constant(Tensor.create(1), "ONE");
    let baseGrad = grad.divide(node.base.square().subtract(one).sqrt());
    return [baseGrad];
  }
}