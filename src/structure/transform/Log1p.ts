import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import ExpressionTypes from "../ExpressionTypes";
import TransformExpression from "./TransformExpression";

export default class Log1p extends TransformExpression {

  get type() {
    return ExpressionTypes.Log1p;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Log1p): Tensor {
    let base = node.base.value;
    return TensorMath.log1p(base);
  }

  static gradients(node: Log1p, grad: Expression): Expression[] {
    let one = node.factory.constant(Tensor.scalar(1), 'ONE');
    let baseGrad = node.base.add(one).reciprocal().multiply(grad);
    return [baseGrad];
  }
}