import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Cosine extends TransformExpression {

  get type() {
    return ExpressionTypes.Cosine;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Cosine;
    let base = node.base.value;
    return TensorMath.cos(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Cosine;
    let baseGrad = node.base.sin().negate().multiply(grad);
    return [baseGrad];
  }
}