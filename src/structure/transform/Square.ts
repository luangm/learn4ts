import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Square extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Square;
  }

  static evaluate(node: Square): Tensor {
    let base = node.base.value;
    return TensorMath.square(base);
  }

  static gradients(node: Square, grad: Expression): Expression[] {
    let two = node.factory.constant(Tensor.scalar(2), 'TWO');
    let baseGrad = two.multiply(node.base).multiply(grad);
    return [baseGrad];
  }
}