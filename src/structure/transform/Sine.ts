import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Sine extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Sine;
  }

  static evaluate(node: Sine): Tensor {
    let base = node.graph.session.getValue(node.base);
    return TensorMath.sin(base);
  }

  static gradients(node: Sine, grad: Expression): Expression[] {
    let cos = node.factory.cos(node.base);
    let result = node.factory.multiply(grad, cos);
    return [result];
  }
}