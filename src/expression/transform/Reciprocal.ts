import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Reciprocal extends TransformExpression {

  get type() {
    return ExpressionTypes.Reciprocal;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Reciprocal;
    let base = node.base.value;
    return TensorMath.reciprocal(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Reciprocal;
    let baseGrad = node.base.reciprocalGrad().multiply(grad);
    return [baseGrad];
  }
}