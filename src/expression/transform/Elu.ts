import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Elu extends TransformExpression {

  get type() {
    return ExpressionTypes.Elu;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Elu): Tensor {
    let base = node.base.value;
    return TensorMath.elu(base);
  }

  static gradients(node: Elu, grad: Expression): Expression[] {
    let baseGrad = node.base.step().multiply(grad);
    return [baseGrad];
  }
}