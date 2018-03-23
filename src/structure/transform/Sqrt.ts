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
    let base = node.base.value;
    return TensorMath.sqrt(base);
  }

  static gradients(node: Sqrt, grad: Expression): Expression[] {
    let baseGrad = node.base.sqrtGrad().multiply(grad);
    return [baseGrad];
  }
}