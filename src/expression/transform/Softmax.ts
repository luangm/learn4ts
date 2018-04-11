import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Softmax extends TransformExpression {

  get type() {
    return ExpressionTypes.Softmax;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Softmax;
    let base = node.base.value;
    return TensorMath.softmax(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Softmax;
    let baseGrad = grad.subtract(grad.multiply(node).reduceSum(1).reshape([-1, 1])).multiply(node);
    return [baseGrad];
  }
}