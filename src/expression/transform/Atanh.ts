import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Atanh extends TransformExpression {

  get type() {
    return ExpressionTypes.Atanh;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Atanh;
    let base = node.base.value;
    return TensorMath.atanh(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Atanh;
    let one = node.factory.constant(Tensor.create(1), "ONE");
    let baseGrad = grad.divide(one.subtract(node.base.square()));
    return [baseGrad];
  }
}