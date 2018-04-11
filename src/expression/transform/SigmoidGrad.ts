import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import Sigmoid from "./Sigmoid";
import TransformExpression from "./TransformExpression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class SigmoidGrad extends TransformExpression {

  get type() {
    return ExpressionTypes.SigmoidGrad;
  }

  constructor(base: Expression, graph: Graph, name?: string) {
    super(base, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as SigmoidGrad;
    let base = node.base.value;
    return TensorMath.sigmoidGrad(base);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Sigmoid;
    let baseGrad = node.base.sigmoidGrad().multiply(grad);
    return [baseGrad];
  }

}