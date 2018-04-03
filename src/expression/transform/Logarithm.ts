import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Logarithm extends TransformExpression {

  get type() {
    return ExpressionTypes.Logarithm;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Logarithm;
    let base = node.base.value;
    return TensorMath.log(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Logarithm;
    let baseGrad = grad.divide(node.base);
    return [baseGrad];
  }
}