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

  static evaluate(expression: Expression): Tensor {
    let node = expression as Elu;
    let base = node.base.value;
    return TensorMath.elu(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Elu;
    let baseGrad = node.base.eluGrad().multiply(grad);
    return [baseGrad];
  }
}