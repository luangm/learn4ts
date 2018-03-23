import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Relu extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Relu;
  }

  static evaluate(node: Relu): Tensor {
    let base = node.base.value;
    return TensorMath.relu(base);
  }

  static gradients(node: Relu, grad: Expression): Expression[] {
    let step = node.factory.step(node.base);
    let result = node.factory.multiply(grad, step);
    return [result];
  }
}