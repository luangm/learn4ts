import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Sigmoid extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Sigmoid;
  }

  static evaluate(node: Sigmoid): Tensor {
    let base = node.base.value;
    return TensorMath.sigmoid(base);
  }

  static gradients(node: Sigmoid, grad: Expression): Expression[] {
    let sigmoidGrad = node.factory.sigmoidGrad(node.base);
    let result = node.factory.multiply(grad, sigmoidGrad);
    return [result];
  }
}