import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Tangent extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Tangent;
  }

  static evaluate(node: Tangent): Tensor {
    let base = node.graph.session.getValue(node.base);
    return TensorMath.tan(base);
  }

  static gradients(node: Tangent, grad: Expression): Expression[] {
    let tanGrad = node.factory.tanGrad(node.base);
    let result = node.factory.multiply(grad, tanGrad);
    return [result];
  }
}