import {Tensor, TensorMath} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import TransformExpression from "../transform/TransformExpression";

export default class Dropout extends TransformExpression {

  private _mask?: Tensor;
  private readonly _probability: number;

  get dependencies() {
    return [this.base];
  }

  get mask() {
    return this._mask;
  }

  set mask(val) {
    this._mask = val;
  }

  get probability() {
    return this._probability;
  }

  get type() {
    return ExpressionTypes.Dropout;
  }

  constructor(base: Expression, probability: number = 0.5, graph: Graph, name?: string) {
    super(base, graph, name);
    this._probability = probability;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Dropout;
    let base = node.base.value;
    node.mask = TensorMath.step(Tensor.rand(base.shape).subtract(Tensor.create(0.5)));
    console.log(node.mask.toString());
    return TensorMath.multiply(base, node.mask);
  }
}