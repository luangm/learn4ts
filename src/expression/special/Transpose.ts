import {ShapeUtils, Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Transpose extends Expression {

  private readonly _base: Expression;
  private readonly _newAxis: number[];
  private readonly _shape: number[];

  get base() {
    return this._base;
  }

  get dependencies(): Expression[] {
    return [this.base];
  }

  get newAxis() {
    return this._newAxis;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      base: this.base.id,
      newAxis: this.newAxis
    };
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Transpose;
  }

  constructor(base: Expression, newAxis: number[] = [], graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
    if (newAxis.length === 0) {
      let rank = base.shape.length;
      newAxis = new Array(rank);
      for (let i = 0; i < newAxis.length; i++) {
        newAxis[i] = rank - 1 - i;
      }
    }
    this._newAxis = newAxis;
    this._shape = ShapeUtils.transposeShape(base.shape, newAxis);
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Transpose;
    let base = node.base.value;
    return base.transpose(node.newAxis);
  }

  static gradients(expression: Expression, grad: Expression): Expression[] {
    let node = expression as Transpose;
    let axis = ShapeUtils.invertPermutation(node.newAxis);
    return [grad.transpose(axis)];
  }
}