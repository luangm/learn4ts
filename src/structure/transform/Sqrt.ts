import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Sqrt extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Sqrt;
  }

  static evaluate(node: Sqrt): Tensor {
    let base = node.graph.session.getValue(node.base);
    return TensorMath.sqrt(base);
  }

  static gradients(node: Sqrt, grad: Expression): Expression[] {
    let sqrtGrad = node.factory.sqrtGrad(node.base);
    let result = node.factory.multiply(grad, sqrtGrad);
    return [result];
  }
}