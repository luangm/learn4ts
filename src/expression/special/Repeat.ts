import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Repeat extends Expression {

  private readonly _base: Expression;
  private readonly _dimension: number;
  private readonly _multiple: number;
  private readonly _shape: number[];

  get base() {
    return this._base;
  }

  get dependencies(): Expression[] {
    return [this.base];
  }

  get dimension() {
    return this._dimension;
  }

  get multiple() {
    return this._multiple;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      base: this.base.id,
      dimension: this.dimension,
      multiple: this.multiple
    };
  }

  get shape() {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.Repeat;
  }

  constructor(base: Expression, multiple: number, dimension: number, graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
    this._multiple = multiple;
    this._dimension = dimension;
    if (dimension === -1) {
      let length = multiple;
      let baseShape = base.shape;
      for (let i = 0; i < baseShape.length; i++) {
        length *= baseShape[i];
      }
      this._shape = [length];
    } else {
      this._shape = base.shape.slice();
      this._shape[dimension] *= multiple;
    }
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Repeat;
    let base = node.base.value;
    return base.repeat(node.multiple, node.dimension);
  }

}