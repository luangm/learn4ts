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

  static evaluate(node: Cosine): Tensor {
    let base = node.base.value;
    return TensorMath.cos(base);
  }

  static gradients(node: Cosine, grad: Expression): Expression[] {
    let baseGrad = node.base.sin().negate().multiply(grad);
    return [baseGrad];
  }
}