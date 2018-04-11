import {ShapeUtils, Tensor, TensorMath} from "tensor4js";
import ReductionExpression from "./ReductionExpression";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class ReduceSum extends ReductionExpression {

  get type() {
    return ExpressionTypes.ReduceSum;
  }

  constructor(base: Expression, dims: number | number[] = -1, keepDims = false, graph: Graph, name?: string) {
    super(base, dims, keepDims, graph, name);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as ReduceSum;
    let base = node.base.value;
    return TensorMath.reduceSum(base, node.dims, node.keepDims);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as ReduceSum;
    let inputShape = node.base.shape;
    let outputShape = ShapeUtils.reduceShape(inputShape, node.dims, true);
    let repeats = ShapeUtils.safeDivide(inputShape, outputShape);
    let baseGrad = grad.reshape(outputShape).tile(repeats);
    return [baseGrad];
  }
}