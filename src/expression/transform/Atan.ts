import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Atan extends TransformExpression {

  get type() {
    return ExpressionTypes.Atan;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Atan;
    let base = node.base.value;
    return TensorMath.atan(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Atan;
    let one = node.factory.constant(Tensor.create(1), "ONE");
    let baseGrad = grad.divide(node.base.square().add(one));
    return [baseGrad];
  }
}