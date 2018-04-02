import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Exponential extends TransformExpression {

  get type() {
    return ExpressionTypes.Exponential;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Exponential): Tensor {
    let base = node.base.value;
    return TensorMath.exp(base);
  }

  static gradients(node: Exponential, grad: Expression): Expression[] {
    let baseGrad = node.multiply(grad);
    return [baseGrad];
  }
}