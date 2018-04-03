import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Relu extends TransformExpression {

  get type() {
    return ExpressionTypes.Relu;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Relu;
    let base = node.base.value;
    return TensorMath.relu(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Relu;
    let baseGrad = node.base.step().multiply(grad);
    return [baseGrad];
  }
}