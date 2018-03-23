import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Logarithm extends TransformExpression {

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  get type() {
    return ExpressionTypes.Logarithm;
  }

  static evaluate(node: Logarithm): Tensor {
    let base = node.base.value;
    return TensorMath.log(base);
  }

  static gradients(node: Logarithm, grad: Expression): Expression[] {
    let result = node.factory.divide(grad, node.base);
    return [result];
  }
}