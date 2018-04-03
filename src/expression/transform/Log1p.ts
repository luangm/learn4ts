import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Log1p extends TransformExpression {

  get type() {
    return ExpressionTypes.Log1p;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Log1p;
    let base = node.base.value;
    return TensorMath.log1p(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Log1p;
    let one = node.factory.constant(Tensor.scalar(1), "ONE");
    let baseGrad = node.base.add(one).reciprocal().multiply(grad);
    return [baseGrad];
  }
}