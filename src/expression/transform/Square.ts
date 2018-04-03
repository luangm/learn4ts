import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Square extends TransformExpression {

  get type() {
    return ExpressionTypes.Square;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Square;
    let base = node.base.value;
    return TensorMath.square(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Square;
    let two = node.factory.constant(Tensor.scalar(2), "TWO");
    let baseGrad = two.multiply(node.base).multiply(grad);
    return [baseGrad];
  }
}