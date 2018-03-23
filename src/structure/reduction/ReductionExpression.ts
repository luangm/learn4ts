import {ShapeUtils} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";

export default abstract class ReductionExpression extends Expression {

  private _base: Expression;
  private _dims: number | number[];
  private _shape: number[];

  constructor(base: Expression, dims: number | number[] = -1, graph: Graph, name?: string) {
    super(graph, name);
    this._base = base;
    this._dims = dims;
    this._shape = ShapeUtils.reduceShape(base.shape, dims, false);
  }

  get base() {
    return this._base;
  }

  get dependencies() {
    return [this.base];
  }

  get dims() {
    return this._dims;
  }

  get shape() {
    return this._shape;
  }

}