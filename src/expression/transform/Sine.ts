import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Sine extends TransformExpression {

  get type() {
    return ExpressionTypes.Sine;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Sine;
    let base = node.base.value;
    return TensorMath.sin(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Sine;
    let baseGrad = node.base.cos().multiply(grad);
    return [baseGrad];
  }
}