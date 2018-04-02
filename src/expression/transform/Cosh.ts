import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Cosh extends TransformExpression {

  get type() {
    return ExpressionTypes.Cosh;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Cosh): Tensor {
    let base = node.base.value;
    return TensorMath.cosh(base);
  }

  static gradients(node: Cosh, grad: Expression): Expression[] {
    let baseGrad = node.base.sinh().multiply(grad);
    return [baseGrad];
  }
}