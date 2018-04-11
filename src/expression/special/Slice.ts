import {Tensor} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export default class Slice extends Expression {

  private readonly _base: Expression;
  private readonly _begin: number[];
  private readonly _shape: number[];
  private readonly _size: number[];

  get base() {
    return this._base;
  }

  get begin() {
    return this._begin;
  }

  get dependencies(): Expression[] {
    return [this.base];
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      base: this.base.id,
      begin: this.begin,
      size: this.size
    };
  }

  get shape() {
    return this._shape;
  }

  get size() {
    return this._size;
  }

  get type() {
    return ExpressionTypes.Slice;
  }

  constructor(base: Expression, begin: number[], size: number[] = [], graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
    this._begin = begin;

    let rank = base.shape.length;
    if (size.length === 0) {
      size = new Array(rank).fill(-1);
    }
    this._size = size;

    let newShape = new Array(rank);
    let shape = base.shape;
    for (let i = 0; i < rank; i++) {
      let a = begin[i] < 0 ? begin[i] + shape[i] : begin[i];
      newShape[i] = size[i] < 0 ? (shape[i] - a) : Math.min(shape[i] - a, size[i]);
    }
    this._shape = newShape;
  }

  static evaluate(expression: Expression): Tensor {
    let node = expression as Slice;
    let base = node.base.value;
    return base.slice(node.begin, node.size);
  }

}