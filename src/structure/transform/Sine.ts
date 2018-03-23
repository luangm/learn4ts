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
    let base = node.base.value;
    return TensorMath.sin(base);
  }

  static gradients(node: Sine, grad: Expression): Expression[] {
    let baseGrad = node.base.cos().multiply(grad);
    return [baseGrad];
  }
}