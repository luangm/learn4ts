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

  static evaluate(node: Reciprocal): Tensor {
    let base = node.base.value;
    return TensorMath.reciprocal(base);
  }

  static gradients(node: Reciprocal, grad: Expression): Expression[] {
    let baseGrad = node.base.reciprocalGrad().multiply(grad);
    return [baseGrad];
  }
}