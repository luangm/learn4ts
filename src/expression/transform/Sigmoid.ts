import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Sigmoid extends TransformExpression {

  get type() {
    return ExpressionTypes.Sigmoid;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(node: Sigmoid): Tensor {
    let base = node.base.value;
    return TensorMath.sigmoid(base);
  }

  static gradients(node: Sigmoid, grad: Expression): Expression[] {
    let baseGrad = node.base.sigmoidGrad().multiply(grad);
    return [baseGrad];
  }
}