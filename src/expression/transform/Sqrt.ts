import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Sqrt extends TransformExpression {

  get type() {
    return ExpressionTypes.Sqrt;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Sqrt;
    let base = node.base.value;
    return TensorMath.sqrt(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Sqrt;
    let baseGrad = node.base.sqrtGrad().multiply(grad);
    return [baseGrad];
  }
}