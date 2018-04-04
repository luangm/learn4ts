import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Asin extends TransformExpression {

  get type() {
    return ExpressionTypes.Asin;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Asin;
    let base = node.base.value;
    return TensorMath.asin(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Asin;
    let one = node.factory.constant(Tensor.create(1), "ONE");
    let baseGrad = grad.divide(one.subtract(node.base.square()).sqrt());
    return [baseGrad];
  }
}