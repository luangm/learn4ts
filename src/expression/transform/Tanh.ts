import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Tanh extends TransformExpression {

  get type() {
    return ExpressionTypes.Tanh;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Tanh): Tensor {
    let base = node.base.value;
    return TensorMath.tanh(base);
  }

  static gradients(node: Tanh, grad: Expression): Expression[] {
    let baseGrad = node.base.tanhGrad().multiply(grad);
    return [baseGrad];
  }
}