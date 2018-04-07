import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Erf extends TransformExpression {

  get type() {
    return ExpressionTypes.Erf;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Erf;
    let base = node.base.value;
    return TensorMath.erf(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Erf;
    let baseGrad = node.base.erfGrad().multiply(grad);
    return [baseGrad];
  }
}