import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Cosine extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Cosine;
  }

  static evaluate(node: Cosine): Tensor {
    let base = node.graph.session.getValue(node.base);
    return TensorMath.cos(base);
  }

  static gradients(node: Cosine, grad: Expression): Expression[] {
    let sine = node.factory.sin(node.base);
    let neg = node.factory.negate(sine);
    let result = node.factory.multiply(grad, neg);
    return [result];
  }
}